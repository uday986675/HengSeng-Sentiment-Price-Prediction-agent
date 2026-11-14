"""Technical indicators and dynamic Support/Resistance level calculators."""
from typing import Dict, Tuple
import numpy as np
import pandas as pd


class TechnicalAnalyzer:
    """Computes technical indicators, moving averages, and dynamic Support/Resistance levels."""

    @staticmethod
    def compute_support_resistance(df: pd.DataFrame, window: int = 14) -> Dict[str, float]:
        """
        Computes dynamic Support and Resistance levels using multiple methodologies:
        1. Classic Floor Trader Pivot Points
        2. Rolling Swing High / Low extrema
        3. Rolling Volume/Price Quantiles (15th and 85th percentiles)
        """
        if df.empty or len(df) < 3:
            return {
                "pivot": 0.0,
                "resistance_1": 0.0,
                "support_1": 0.0,
                "resistance_2": 0.0,
                "support_2": 0.0,
                "swing_high": 0.0,
                "swing_low": 0.0,
                "quantile_support": 0.0,
                "quantile_resistance": 0.0,
            }

        last_row = df.iloc[-1]
        high = float(last_row["high"])
        low = float(last_row["low"])
        close = float(last_row["close"])

        # 1. Classic Pivot Points
        pivot = (high + low + close) / 3.0
        r1 = (2.0 * pivot) - low
        s1 = (2.0 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)

        # 2. Rolling Swing Extrema
        sub_df = df.tail(min(len(df), window))
        swing_high = float(sub_df["high"].max())
        swing_low = float(sub_df["low"].min())

        # 3. Rolling Quantile Levels
        quantile_support = float(sub_df["close"].quantile(0.15))
        quantile_resistance = float(sub_df["close"].quantile(0.85))

        return {
            "pivot": round(pivot, 2),
            "resistance_1": round(r1, 2),
            "support_1": round(s1, 2),
            "resistance_2": round(r2, 2),
            "support_2": round(s2, 2),
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
            "quantile_support": round(quantile_support, 2),
            "quantile_resistance": round(quantile_resistance, 2),
        }

    @staticmethod
    def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Enriches price DataFrame with momentum, volatility, and trend indicators."""
        df = df.copy().sort_values("timestamp").reset_index(drop=True)

        # Daily Return
        df["return_1d"] = df["close"].pct_change(1)
        df["return_3d"] = df["close"].pct_change(3)
        df["return_5d"] = df["close"].pct_change(5)

        # Moving averages
        df["sma_5"] = df["close"].rolling(5, min_periods=1).mean()
        df["sma_20"] = df["close"].rolling(20, min_periods=1).mean()
        df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()

        # Trend ratio: close to 20-day SMA
        df["price_to_sma20"] = df["close"] / (df["sma_20"] + 1e-8)

        # Volatility: 10-day rolling standard deviation of returns
        df["volatility_10d"] = df["return_1d"].rolling(10, min_periods=1).std().fillna(0.015)

        # Relative Strength Index (RSI 14)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Fill NaNs
        df = df.bfill().ffill()
        return df
