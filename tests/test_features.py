"""Unit tests for sentiment scoring, technical analysis, and feature engineering."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from agent.features.sentiment_analyzer import FinancialSentimentAnalyzer
from agent.features.technicals import TechnicalAnalyzer
from agent.features.feature_engineering import FeatureEngineer


def test_multilingual_sentiment_scoring():
    analyzer = FinancialSentimentAnalyzer()
    
    # English Bullish & Bearish
    score_en_bull, dir_en_bull = analyzer.score_text("Strong earnings beat and accelerated share buyback announce bullish breakout!")
    assert score_en_bull > 0
    assert dir_en_bull == "bullish"

    score_en_bear, dir_en_bear = analyzer.score_text("Regulatory probe causes massive selloff and dividend cut with severe loss.")
    assert score_en_bear < 0
    assert dir_en_bear == "bearish"

    # Chinese Bullish & Bearish (HKEX / LIHKG / Futu)
    score_zh_bull, dir_zh_bull = analyzer.score_text("騰訊南向資金大買，WeChat視頻號電商GMV翻倍，明天必升破400關口！🚀")
    assert score_zh_bull > 0
    assert dir_zh_bull == "bullish"

    score_zh_bear, dir_zh_bear = analyzer.score_text("大戶疑似散貨，追高小心套牢，沽空比率飆升，睇跌破位。")
    assert score_zh_bear < 0
    assert dir_zh_bear == "bearish"

    # Korean Bullish & Bearish (KRX / Naver Finance)
    score_kr_bull, dir_kr_bull = analyzer.score_text("삼성전자 드디어 HBM3E 엔비디아 승인 임박! 영업이익 대폭 흑자 턴어라운드!")
    assert score_kr_bull > 0
    assert dir_kr_bull == "bullish"

    score_kr_bear, dir_kr_bear = analyzer.score_text("파운드리 적자 지속 및 반도체 재고 부담으로 급락 우려.")
    assert score_kr_bear < 0
    assert dir_kr_bear == "bearish"


def test_technical_analyzer_support_resistance():
    data = []
    base_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(20):
        data.append({
            "timestamp": base_date + timedelta(days=i),
            "open": 100.0 + i,
            "high": 105.0 + i,
            "low": 98.0 + i,
            "close": 103.0 + i,
            "volume": 1000000,
            "adjusted_close": 103.0 + i
        })
    df = pd.DataFrame(data)
    
    sr = TechnicalAnalyzer.compute_support_resistance(df, window=10)
    assert "pivot" in sr
    assert "resistance_1" in sr
    assert "support_1" in sr
    assert sr["resistance_1"] > sr["support_1"]
    assert sr["swing_high"] >= sr["swing_low"]

    enriched = TechnicalAnalyzer.add_technical_features(df)
    assert "return_1d" in enriched.columns
    assert "rsi_14" in enriched.columns
    assert "sma_5" in enriched.columns


def test_feature_engineer_metrics():
    engineer = FeatureEngineer()
    
    base_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
    prices = pd.DataFrame([
        {"ticker": "0700.HK", "timestamp": base_date + timedelta(days=i), "open": 380 + i, "high": 385 + i, "low": 375 + i, "close": 382 + i, "volume": 500000, "adjusted_close": 382 + i}
        for i in range(15)
    ])
    
    social = pd.DataFrame([
        {
            "post_id": f"p_{i}",
            "ticker": "0700.HK",
            "platform": "telegram",
            "timestamp": base_date + timedelta(days=i, hours=10),
            "text": "Tencent surge revenue growth beating expectations! 🚀",
            "author": "analyst",
            "author_followers": 5000,
            "likes": 50,
            "reposts": 10,
            "views": 1000,
            "total_engagement": 220,
            "url": "https://example.com"
        }
        for i in range(15)
    ])

    news = pd.DataFrame([
        {
            "article_id": f"n_{i}",
            "ticker": "0700.HK",
            "source_engine": "exa",
            "timestamp": base_date + timedelta(days=i, hours=4),
            "title": "Tencent Expands Enterprise Cloud Footprint",
            "snippet": "Strong quarterly performance and expanding operating margins.",
            "url": "https://example.com",
            "publisher": "Financial Times",
            "relevance_score": 0.95
        }
        for i in range(15)
    ])

    ml_df, summaries = engineer.process_all_tickers(
        social_df=social,
        news_df=news,
        prices_df=prices,
        asset_metadata={"0700.HK": "Tencent Holdings"},
        currency="HKD"
    )

    assert not ml_df.empty
    assert len(summaries) == 1
    summary = summaries[0]
    
    assert summary.ticker == "0700.HK"
    assert summary.sentiment_direction == "bullish"
    assert "sentiment_velocity" in ml_df.columns
    assert "mention_volume_zscore" in ml_df.columns
    assert "sentiment_momentum" in ml_df.columns
    assert "news_social_divergence" in ml_df.columns
    assert summary.suggested_position in ["LONG", "SHORT", "NEUTRAL"]

