"""Quantitative feature engineering engine combining sentiment, engagement, and price signals."""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from agent.features.sentiment_analyzer import FinancialSentimentAnalyzer
from agent.features.technicals import TechnicalAnalyzer
from agent.utils.logger import setup_logger

logger = setup_logger("FeatureEngineering")


class TickerExtractionSummary(BaseModel):
    """Encapsulates the required per-ticker extraction payload."""
    ticker: str
    asset_name: str
    current_price: float
    currency: str
    aggregate_sentiment_score: float
    sentiment_direction: str # bullish | bearish | neutral
    social_sentiment_score: float
    news_sentiment_score: float
    sentiment_velocity: float
    mention_volume_zscore: float
    sentiment_momentum: float
    news_social_divergence: float
    total_social_mentions_10d: int
    total_news_articles_10d: int
    suggested_position: str # LONG | SHORT | NEUTRAL
    position_rationale: str
    support_resistance: Dict[str, float]
    latest_headlines: List[str] = Field(default_factory=list)


class FeatureEngineer:
    """Transforms raw social posts, news articles, and price bars into predictive ML feature matrices."""

    def __init__(self, sentiment_decay_halflife_hours: float = 24.0):
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        self.halflife_hours = sentiment_decay_halflife_hours

    def process_all_tickers(
        self,
        social_df: pd.DataFrame,
        news_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        asset_metadata: Dict[str, str], # ticker -> asset_name
        currency: str = "HKD"
    ) -> Tuple[pd.DataFrame, List[TickerExtractionSummary]]:
        """
        Processes all tickers, creates unified ML training dataset with strict temporal alignment,
        and generates extraction summaries per ticker.
        """
        all_features_dfs = []
        summaries: List[TickerExtractionSummary] = []

        # Group by ticker
        tickers = list(prices_df["ticker"].unique())

        for ticker in tickers:
            asset_name = asset_metadata.get(ticker, ticker)
            t_prices = prices_df[prices_df["ticker"] == ticker].copy().sort_values("timestamp").reset_index(drop=True)
            t_social = social_df[social_df["ticker"] == ticker].copy() if not social_df.empty else pd.DataFrame()
            t_news = news_df[news_df["ticker"] == ticker].copy() if not news_df.empty else pd.DataFrame()

            if len(t_prices) < 5:
                logger.warning(f"Insufficient price history for {ticker} ({len(t_prices)} bars). Skipping.")
                continue

            # Build daily features
            df_feat, summary = self._build_ticker_features(
                ticker=ticker,
                asset_name=asset_name,
                t_prices=t_prices,
                t_social=t_social,
                t_news=t_news,
                currency=currency
            )

            if not df_feat.empty:
                all_features_dfs.append(df_feat)
                summaries.append(summary)

        combined_ml_df = pd.concat(all_features_dfs, ignore_index=True) if all_features_dfs else pd.DataFrame()
        return combined_ml_df, summaries

    def _build_ticker_features(
        self,
        ticker: str,
        asset_name: str,
        t_prices: pd.DataFrame,
        t_social: pd.DataFrame,
        t_news: pd.DataFrame,
        currency: str
    ) -> Tuple[pd.DataFrame, TickerExtractionSummary]:
        """Constructs time-series feature matrix and summary for a single ticker."""
        # 1. Technical indicators
        t_prices = TechnicalAnalyzer.add_technical_features(t_prices)

        # 2. Score social posts if not already scored
        if not t_social.empty:
            t_social["sentiment_score"] = t_social["text"].apply(lambda txt: self.sentiment_analyzer.score_text(txt)[0])
            t_social["post_date"] = pd.to_datetime(t_social["timestamp"], utc=True).dt.floor("D")
            # Engagement weight: log(1 + engagement) * log(10 + followers)
            t_social["weight"] = np.log1p(t_social["total_engagement"].clip(lower=0)) * np.log1p(t_social["author_followers"].clip(lower=0) + 10)
        else:
            t_social["sentiment_score"] = 0.0
            t_social["post_date"] = pd.to_datetime([], utc=True)
            t_social["weight"] = 1.0

        # 3. Score news articles
        if not t_news.empty:
            t_news["sentiment_score"] = t_news.apply(
                lambda row: self.sentiment_analyzer.score_text(f"{row['title']} {row.get('snippet', '')}")[0], axis=1
            )
            t_news["news_date"] = pd.to_datetime(t_news["timestamp"], utc=True).dt.floor("D")
        else:
            t_news["sentiment_score"] = 0.0
            t_news["news_date"] = pd.to_datetime([], utc=True)

        # 4. Map daily aggregated signals to each price bar date
        rows = []
        for i, row in t_prices.iterrows():
            bar_date = row["timestamp"] # pd.Timestamp at 00:00:00 UTC

            # Daily social window: up to bar date
            sub_social = t_social[t_social["post_date"] == bar_date]
            if sub_social.empty:
                # Lookback 24-48 hours window
                sub_social = t_social[(t_social["post_date"] <= bar_date) & (t_social["post_date"] >= bar_date - pd.Timedelta(days=2))]

            mention_count = len(sub_social)
            if mention_count > 0 and sub_social["weight"].sum() > 0:
                weighted_social_sent = float((sub_social["sentiment_score"] * sub_social["weight"]).sum() / sub_social["weight"].sum())
                total_eng = float(sub_social["total_engagement"].sum())
            else:
                weighted_social_sent = 0.0
                total_eng = 0.0

            # Daily news window
            sub_news = t_news[t_news["news_date"] == bar_date]
            if sub_news.empty:
                sub_news = t_news[(t_news["news_date"] <= bar_date) & (t_news["news_date"] >= bar_date - pd.Timedelta(days=3))]

            news_count = len(sub_news)
            if news_count > 0:
                news_sent = float(sub_news["sentiment_score"].mean())
            else:
                news_sent = 0.0

            # Composite sentiment: 60% engagement-weighted social + 40% news
            if mention_count > 0 and news_count > 0:
                composite_sent = 0.6 * weighted_social_sent + 0.4 * news_sent
            elif mention_count > 0:
                composite_sent = weighted_social_sent
            elif news_count > 0:
                composite_sent = news_sent
            else:
                composite_sent = 0.0

            rows.append({
                "ticker": ticker,
                "timestamp": bar_date,
                "close": row["close"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "volume": row["volume"],
                "return_1d": row["return_1d"],
                "return_3d": row["return_3d"],
                "return_5d": row["return_5d"],
                "sma_5": row["sma_5"],
                "sma_20": row["sma_20"],
                "price_to_sma20": row["price_to_sma20"],
                "volatility_10d": row["volatility_10d"],
                "rsi_14": row["rsi_14"],
                "social_mentions": mention_count,
                "social_engagement": total_eng,
                "social_sentiment": weighted_social_sent,
                "news_count": news_count,
                "news_sentiment": news_sent,
                "composite_sentiment": composite_sent,
            })

        df_out = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

        # 5. Creative Feature Engineering across the time series:
        # A. Engagement-weighted sentiment velocity: dS/dt (1-day diff)
        df_out["sentiment_velocity"] = df_out["composite_sentiment"].diff(1).fillna(0.0)

        # B. Mention-Volume Z-Score: (Vol - Mean_7d) / Std_7d
        vol_mean = df_out["social_mentions"].rolling(7, min_periods=1).mean()
        vol_std = df_out["social_mentions"].rolling(7, min_periods=1).std().fillna(1.0)
        df_out["mention_volume_zscore"] = (df_out["social_mentions"] - vol_mean) / (vol_std + 1e-4)

        # C. Sentiment Momentum: fast EMA (3-day) minus slow EMA (7-day)
        fast_ema = df_out["composite_sentiment"].ewm(span=3, adjust=False).mean()
        slow_ema = df_out["composite_sentiment"].ewm(span=7, adjust=False).mean()
        df_out["sentiment_momentum"] = fast_ema - slow_ema

        # D. News-vs-Social Divergence: News Sentiment minus Social Sentiment
        df_out["news_social_divergence"] = df_out["news_sentiment"] - df_out["social_sentiment"]

        # E. Future targets at n+1 (for supervised training without lookahead)
        # Shift future close back by 1 row
        df_out["target_next_close"] = df_out["close"].shift(-1)
        df_out["target_next_return"] = (df_out["target_next_close"] - df_out["close"]) / df_out["close"]
        df_out["target_next_direction"] = np.where(df_out["target_next_return"] > 0, 1, 0)

        # 6. Compute Support / Resistance for latest bar
        sr_levels = TechnicalAnalyzer.compute_support_resistance(t_prices)

        # 7. Generate Current Ticker Summary
        latest_row = df_out.iloc[-1]
        current_price = float(latest_row["close"])
        latest_comp_sent = float(latest_row["composite_sentiment"])
        latest_vel = float(latest_row["sentiment_velocity"])
        latest_zscore = float(latest_row["mention_volume_zscore"])
        latest_mom = float(latest_row["sentiment_momentum"])
        latest_div = float(latest_row["news_social_divergence"])

        if latest_comp_sent >= 0.15:
            sent_direction = "bullish"
        elif latest_comp_sent <= -0.15:
            sent_direction = "bearish"
        else:
            sent_direction = "neutral"

        # Suggested position logic:
        # Cross sentiment momentum and velocity with price relative to pivot
        pivot_val = sr_levels.get("pivot", current_price)
        if (latest_comp_sent > 0.10 and latest_vel >= -0.05 and current_price >= pivot_val * 0.99) or (latest_comp_sent > 0.25):
            suggested_pos = "LONG"
            rationale = (f"Bullish composite sentiment ({latest_comp_sent:+.2f}) accompanied by positive momentum "
                         f"({latest_mom:+.2f}) and support above key pivot ({pivot_val:.2f} {currency}).")
        elif (latest_comp_sent < -0.10 and latest_vel <= 0.05 and current_price <= pivot_val * 1.01) or (latest_comp_sent < -0.25):
            suggested_pos = "SHORT"
            rationale = (f"Bearish crowd bias ({latest_comp_sent:+.2f}) and negative sentiment velocity "
                         f"({latest_vel:+.2f}) pressing below resistance ({sr_levels.get('resistance_1', 0.0):.2f} {currency}).")
        else:
            suggested_pos = "NEUTRAL"
            rationale = (f"Balanced sentiment profile ({latest_comp_sent:+.2f}); price consolidating within S/R range "
                         f"[{sr_levels.get('support_1', 0.0):.2f}, {sr_levels.get('resistance_1', 0.0):.2f}].")

        # Extract latest headlines
        latest_headlines = []
        if not t_news.empty:
            recent_sorted = t_news.sort_values("timestamp", ascending=False).head(3)
            latest_headlines = recent_sorted["title"].tolist()

        summary = TickerExtractionSummary(
            ticker=ticker,
            asset_name=asset_name,
            current_price=round(current_price, 2),
            currency=currency,
            aggregate_sentiment_score=round(latest_comp_sent, 4),
            sentiment_direction=sent_direction,
            social_sentiment_score=round(float(latest_row["social_sentiment"]), 4),
            news_sentiment_score=round(float(latest_row["news_sentiment"]), 4),
            sentiment_velocity=round(latest_vel, 4),
            mention_volume_zscore=round(latest_zscore, 2),
            sentiment_momentum=round(latest_mom, 4),
            news_social_divergence=round(latest_div, 4),
            total_social_mentions_10d=int(df_out["social_mentions"].tail(10).sum()),
            total_news_articles_10d=int(df_out["news_count"].tail(10).sum()),
            suggested_position=suggested_pos,
            position_rationale=rationale,
            support_resistance=sr_levels,
            latest_headlines=latest_headlines
        )

        return df_out, summary
