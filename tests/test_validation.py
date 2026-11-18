"""Unit tests for walk-forward validation splitter, baselines, and mock clients."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from agent.modeling.validation import WalkForwardTimeSeriesSplit, NaiveLastPriceBaseline, DriftBaseline, calculate_metrics
from agent.ingestion.apify_client import ApifySocialScraper
from agent.ingestion.exa_client import ExaNewsClient


def test_walk_forward_temporal_ordering():
    df = pd.DataFrame({"x": np.arange(50)})
    splitter = WalkForwardTimeSeriesSplit(n_splits=4, min_train_size=20, test_size=5)
    
    splits = list(splitter.split(df))
    assert len(splits) == 4

    prev_train_end = 0
    for train_idx, test_idx in splits:
        # Strict temporal ordering: all train points occur before test points
        assert np.max(train_idx) < np.min(test_idx), "Lookahead violation: train index >= test index!"
        
        # Test index is contiguous
        assert np.all(np.diff(test_idx) == 1)
        
        # Expanding window: train size must increase or stay same
        assert len(train_idx) >= prev_train_end
        prev_train_end = len(train_idx)


def test_naive_and_drift_baselines():
    current_prices = np.array([100.0, 200.0, 300.0])
    
    # Naive Last Price
    naive = NaiveLastPriceBaseline()
    pred_naive = naive.predict_price(current_prices)
    np.testing.assert_array_equal(pred_naive, current_prices)

    # Drift Baseline
    hist_returns = np.array([0.01, 0.02, 0.03]) # mean = 0.02
    drift = DriftBaseline().fit(hist_returns)
    pred_drift = drift.predict_price(current_prices)
    np.testing.assert_allclose(pred_drift, current_prices * 1.02)


def test_metric_calculations():
    y_true = np.array([105.0, 95.0, 100.0])
    y_pred = np.array([104.0, 96.0, 101.0])
    current = np.array([100.0, 100.0, 100.0])
    
    metrics = calculate_metrics(y_true, y_pred, current)
    assert metrics["MAE"] == 1.0
    assert metrics["RMSE"] == 1.0
    assert metrics["Directional_Accuracy_pct"] == 100.0


def test_apify_mock_client():
    mock_scraper = ApifySocialScraper(api_token="test_token", use_fallback=True)
    # Even if external token fails, fallback returns high-fidelity posts
    posts = mock_scraper.fetch_posts(
        ticker="0700.HK",
        asset_name="Tencent Holdings",
        keywords=["Tencent"],
        cashtags=["#0700"],
        days_back=5
    )
    assert len(posts) > 0
    assert posts[0].ticker == "0700.HK"
    assert hasattr(posts[0], "total_engagement")


def test_exa_mock_client():
    mock_news = ExaNewsClient(api_key="test_key", use_fallback=True)
    articles = mock_news.fetch_news(
        ticker="0700.HK",
        asset_name="Tencent Holdings",
        keywords=["Tencent"],
        days_back=5
    )
    assert len(articles) > 0
    assert articles[0].ticker == "0700.HK"
    assert hasattr(articles[0], "relevance_score")

