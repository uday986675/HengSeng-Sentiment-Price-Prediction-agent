"""Unit tests for UTC timestamp alignment and zero-lookahead target creation."""
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from agent.features.feature_engineering import FeatureEngineer


def test_utc_alignment_and_no_lookahead():
    engineer = FeatureEngineer()
    
    # 10 business days
    base_date = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
    dates = [base_date + timedelta(days=i) for i in range(10)]
    prices_list = [100.0, 102.0, 101.0, 105.0, 104.0, 108.0, 107.0, 110.0, 112.0, 115.0]

    prices = pd.DataFrame([
        {
            "ticker": "9988.HK",
            "timestamp": dates[i],
            "open": prices_list[i] - 0.5,
            "high": prices_list[i] + 1.0,
            "low": prices_list[i] - 1.0,
            "close": prices_list[i],
            "volume": 2000000,
            "adjusted_close": prices_list[i]
        }
        for i in range(10)
    ])

    # Social posts at different UTC hours within each day
    social = pd.DataFrame([
        {
            "post_id": f"p_{i}",
            "ticker": "9988.HK",
            "platform": "lihkg",
            "timestamp": dates[i] + timedelta(hours=6),
            "text": "Alibaba ecommerce order growth solid.",
            "author": "trader",
            "author_followers": 100,
            "likes": 10,
            "reposts": 2,
            "views": 100,
            "total_engagement": 30,
            "url": "https://lihkg.com/post/1"
        }
        for i in range(10)
    ])

    news = pd.DataFrame([
        {
            "article_id": f"n_{i}",
            "ticker": "9988.HK",
            "source_engine": "exa",
            "timestamp": dates[i] + timedelta(hours=2),
            "title": "Alibaba Cloud Innovations",
            "snippet": "Strong cloud revenue growth.",
            "url": "https://news.com/1",
            "publisher": "Reuters",
            "relevance_score": 0.9
        }
        for i in range(10)
    ])

    ml_df, summaries = engineer.process_all_tickers(
        social_df=social,
        news_df=news,
        prices_df=prices,
        asset_metadata={"9988.HK": "Alibaba Group"},
        currency="HKD"
    )

    # 1. Check UTC timestamps
    assert pd.to_datetime(ml_df["timestamp"]).dt.tz is not None
    
    # 2. Check target_next_close strictly matches next day's close
    for i in range(len(ml_df) - 1):
        curr_row = ml_df.iloc[i]
        next_row = ml_df.iloc[i + 1]
        assert curr_row["target_next_close"] == next_row["close"], (
            f"Row {i} target_next_close ({curr_row['target_next_close']}) "
            f"does not match row {i+1} close ({next_row['close']})"
        )

    # 3. Last row must have NaN for target_next_close because future is unknown (no lookahead!)
    assert pd.isna(ml_df.iloc[-1]["target_next_close"])
