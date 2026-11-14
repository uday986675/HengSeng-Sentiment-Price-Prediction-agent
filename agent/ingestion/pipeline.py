"""Data ingestion pipeline orchestrating social scrapers, Exa news, and price feeds."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from agent.config.settings import PipelineConfig, ExchangeConfig
from agent.ingestion.base import SocialPost, NewsArticle, PriceBar
from agent.ingestion.apify_client import ApifySocialScraper
from agent.ingestion.exa_client import ExaNewsClient
from agent.ingestion.tavily_client import TavilyIngestionClient
from agent.ingestion.price_client import YFinancePriceClient
from agent.utils.logger import setup_logger

logger = setup_logger("IngestionPipeline")


class IngestionPipeline:
    """Orchestrates multi-source sentiment scraping and market price data ingestion."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.exchange_config: ExchangeConfig = config.exchanges[config.exchange]
        
        # Directories
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path(config.processed_dir)
        self.cache_dir = Path(config.cache_dir)
        for d in [self.raw_dir, self.processed_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Clients
        self.price_client = YFinancePriceClient(cache_dir=self.cache_dir / "prices")
        self.exa_client = ExaNewsClient(api_key=config.exa_api_key, use_fallback=config.use_synthetic_fallback)
        self.tavily_client = TavilyIngestionClient(api_key=config.tavily_api_key, use_fallback=config.use_synthetic_fallback)
        
        # Setup social scrapers for the exchange (at least 2 distinct sources)
        self.social_scrapers: List[Tuple[str, ApifySocialScraper]] = []
        for src in self.exchange_config.social_sources:
            if src.enabled:
                scraper = ApifySocialScraper(
                    api_token=config.apify_token,
                    use_fallback=config.use_synthetic_fallback,
                    actor_id=src.actor_id
                )
                self.social_scrapers.append((src.name, scraper))

    def run(self, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes end-to-end ingestion across all assets in the active exchange.
        Returns:
            (social_df, news_df, prices_df)
        """
        social_cache_file = self.processed_dir / f"{self.config.exchange}_social_posts.parquet"
        news_cache_file = self.processed_dir / f"{self.config.exchange}_news_articles.parquet"
        prices_cache_file = self.processed_dir / f"{self.config.exchange}_prices.parquet"

        if not force_refresh and social_cache_file.exists() and news_cache_file.exists() and prices_cache_file.exists():
            logger.info("Found cached datasets in data/processed/. Loading cached data...")
            social_df = pd.read_parquet(social_cache_file)
            news_df = pd.read_parquet(news_cache_file)
            prices_df = pd.read_parquet(prices_cache_file)
            return social_df, news_df, prices_df

        logger.info(f"Initiating ingestion for exchange '{self.config.exchange.upper()}' ({len(self.exchange_config.assets)} assets)...")

        all_posts: List[SocialPost] = []
        all_news: List[NewsArticle] = []
        all_prices: List[PriceBar] = []

        now_utc = datetime.now(timezone.utc)
        start_date = now_utc - timedelta(days=self.config.history_days)

        for asset in self.exchange_config.assets:
            logger.info(f"--- Processing {asset.ticker} ({asset.name}) ---")
            
            # 1. Scrape social channels from multiple distinct sources
            asset_posts: List[SocialPost] = []
            distinct_sources_used = set()

            for src_name, scraper in self.social_scrapers:
                try:
                    posts = scraper.fetch_posts(
                        ticker=asset.ticker,
                        asset_name=asset.name,
                        keywords=asset.keywords,
                        cashtags=asset.cashtags,
                        days_back=self.config.sentiment_lookback_days
                    )
                    if posts:
                        asset_posts.extend(posts)
                        distinct_sources_used.add(src_name)
                except Exception as e:
                    logger.warning(f"Error scraping {src_name} for {asset.ticker}: {e}")

            # If fewer than 2 distinct sources succeeded, augment with Tavily social search
            if len(distinct_sources_used) < 2:
                logger.info(f"Less than 2 distinct social sources succeeded ({len(distinct_sources_used)}). Activating Tavily fallback...")
                tavily_posts = self.tavily_client.fetch_posts(
                    ticker=asset.ticker,
                    asset_name=asset.name,
                    keywords=asset.keywords,
                    cashtags=asset.cashtags,
                    days_back=self.config.sentiment_lookback_days
                )
                asset_posts.extend(tavily_posts)
                distinct_sources_used.add("tavily_community")

            logger.info(f"Collected {len(asset_posts)} posts from {len(distinct_sources_used)} distinct sources for {asset.ticker}.")
            all_posts.extend(asset_posts)

            # 2. Ingest news articles via Exa
            try:
                news = self.exa_client.fetch_news(
                    ticker=asset.ticker,
                    asset_name=asset.name,
                    keywords=asset.keywords,
                    days_back=self.config.sentiment_lookback_days,
                    max_results=self.config.exa.num_results_per_asset
                )
                if not news:
                    # Fallback to Tavily
                    news = self.tavily_client.fetch_news(
                        ticker=asset.ticker,
                        asset_name=asset.name,
                        keywords=asset.keywords,
                        days_back=self.config.sentiment_lookback_days
                    )
                all_news.extend(news)
            except Exception as e:
                logger.warning(f"Failed to fetch news for {asset.ticker}: {e}")

            # 3. Ingest market price bars
            try:
                prices = self.price_client.fetch_prices(
                    ticker=asset.ticker,
                    start_date=start_date,
                    end_date=now_utc
                )
                all_prices.extend(prices)
            except Exception as e:
                logger.error(f"Failed to fetch prices for {asset.ticker}: {e}")

        # Convert to standardized DataFrames and Deduplicate
        social_df = self._posts_to_dataframe(all_posts)
        news_df = self._news_to_dataframe(all_news)
        prices_df = self._prices_to_dataframe(all_prices)

        # Save to Parquet
        social_df.to_parquet(social_cache_file, index=False)
        news_df.to_parquet(news_cache_file, index=False)
        prices_df.to_parquet(prices_cache_file, index=False)

        logger.info(f"Ingestion complete. Saved datasets: "
                    f"Social={len(social_df)}, News={len(news_df)}, Prices={len(prices_df)}.")
        return social_df, news_df, prices_df

    def _posts_to_dataframe(self, posts: List[SocialPost]) -> pd.DataFrame:
        """Converts social posts to deduplicated DataFrame with UTC timestamp."""
        if not posts:
            return pd.DataFrame(columns=[
                "post_id", "ticker", "platform", "timestamp", "text",
                "author", "author_followers", "likes", "reposts", "views",
                "total_engagement", "url"
            ])
        data = []
        for p in posts:
            data.append({
                "post_id": p.post_id,
                "ticker": p.ticker,
                "platform": p.platform,
                "timestamp": pd.to_datetime(p.timestamp, utc=True),
                "text": p.text,
                "author": p.author,
                "author_followers": p.author_followers,
                "likes": p.likes,
                "reposts": p.reposts,
                "views": p.views,
                "total_engagement": p.total_engagement,
                "url": p.url
            })
        df = pd.DataFrame(data)
        # Deduplicate by (ticker, text) or post_id
        df = df.drop_duplicates(subset=["post_id"]).drop_duplicates(subset=["ticker", "text"])
        df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
        return df

    def _news_to_dataframe(self, news: List[NewsArticle]) -> pd.DataFrame:
        """Converts news articles to deduplicated DataFrame with UTC timestamp."""
        if not news:
            return pd.DataFrame(columns=[
                "article_id", "ticker", "source_engine", "timestamp",
                "title", "snippet", "url", "publisher", "relevance_score"
            ])
        data = []
        for n in news:
            data.append({
                "article_id": n.article_id,
                "ticker": n.ticker,
                "source_engine": n.source_engine,
                "timestamp": pd.to_datetime(n.timestamp, utc=True),
                "title": n.title,
                "snippet": n.snippet,
                "url": n.url,
                "publisher": n.publisher,
                "relevance_score": n.relevance_score
            })
        df = pd.DataFrame(data)
        df = df.drop_duplicates(subset=["article_id"]).drop_duplicates(subset=["ticker", "title"])
        df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
        return df

    def _prices_to_dataframe(self, prices: List[PriceBar]) -> pd.DataFrame:
        """Converts price bars to sorted DataFrame with UTC date normalization."""
        if not prices:
            return pd.DataFrame(columns=[
                "ticker", "timestamp", "open", "high", "low", "close", "volume", "adjusted_close"
            ])
        data = []
        for p in prices:
            data.append({
                "ticker": p.ticker,
                "timestamp": pd.to_datetime(p.timestamp, utc=True).floor("D"), # daily bar alignment
                "open": p.open,
                "high": p.high,
                "low": p.low,
                "close": p.close,
                "volume": p.volume,
                "adjusted_close": p.adjusted_close
            })
        df = pd.DataFrame(data)
        df = df.drop_duplicates(subset=["ticker", "timestamp"])
        df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
        return df
