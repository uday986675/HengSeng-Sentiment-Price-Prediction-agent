"""Predictive modeling engine for n+1 asset price and return forecasting."""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from agent.utils.logger import setup_logger

logger = setup_logger("PricePredictor")

FEATURE_COLUMNS = [
    # Technical momentum & volatility
    "return_1d", "return_3d", "return_5d",
    "rsi_14", "price_to_sma20", "volatility_10d",
    # Alternative Sentiment & Volume Signals
    "social_mentions", "social_engagement", "social_sentiment",
    "news_count", "news_sentiment", "composite_sentiment",
    # Creative Quantitative Alpha Signals
    "sentiment_velocity", "mention_volume_zscore",
    "sentiment_momentum", "news_social_divergence"
]


class SentimentPricePredictor:
    """Predicts next-bar return R_{n+1} and reconstructs price P_{n+1} = P_n * (1 + R_{n+1})."""

    def __init__(self, model_type: str = "ensemble", random_state: int = 42):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_columns = FEATURE_COLUMNS
        
        # Instantiate base models
        self.ridge = Ridge(alpha=1.0, random_state=random_state)
        self.rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=random_state)
        self.gb = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.05, random_state=random_state)
        
        self.is_fitted = False
        self.feature_importances_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y_return: np.ndarray) -> "SentimentPricePredictor":
        """Fits model on training features and stationary next-bar returns."""
        X_mat = self._prepare_X(X)
        y_vec = np.asarray(y_return, dtype=float)
        
        # Mask out any NaNs
        valid_mask = ~np.isnan(y_vec) & ~np.isnan(X_mat).any(axis=1)
        X_clean = X_mat[valid_mask]
        y_clean = y_vec[valid_mask]

        if len(y_clean) < 5:
            logger.warning("Fewer than 5 valid training samples. Fitting minimal estimator.")
            self.is_fitted = True
            return self

        X_scaled = self.scaler.fit_transform(X_clean)

        if self.model_type == "ridge":
            self.ridge.fit(X_scaled, y_clean)
            importances = np.abs(self.ridge.coef_)
        elif self.model_type == "random_forest":
            self.rf.fit(X_scaled, y_clean)
            importances = self.rf.feature_importances_
        elif self.model_type == "gradient_boosting":
            self.gb.fit(X_scaled, y_clean)
            importances = self.gb.feature_importances_
        else: # "ensemble"
            self.ridge.fit(X_scaled, y_clean)
            self.rf.fit(X_scaled, y_clean)
            self.gb.fit(X_scaled, y_clean)
            importances = 0.3 * np.abs(self.ridge.coef_) + 0.35 * self.rf.feature_importances_ + 0.35 * self.gb.feature_importances_

        total_imp = np.sum(importances) + 1e-8
        norm_imp = importances / total_imp
        self.feature_importances_ = {col: round(float(imp), 4) for col, imp in zip(self.feature_columns, norm_imp)}

        self.is_fitted = True
        return self

    def predict_return(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts stationary next-bar return R_{n+1}."""
        if not self.is_fitted:
            return np.zeros(len(X))

        X_mat = self._prepare_X(X)
        X_scaled = self.scaler.transform(X_mat)

        if self.model_type == "ridge":
            preds = self.ridge.predict(X_scaled)
        elif self.model_type == "random_forest":
            preds = self.rf.predict(X_scaled)
        elif self.model_type == "gradient_boosting":
            preds = self.gb.predict(X_scaled)
        else:
            p_ridge = self.ridge.predict(X_scaled)
            p_rf = self.rf.predict(X_scaled)
            p_gb = self.gb.predict(X_scaled)
            preds = 0.30 * p_ridge + 0.35 * p_rf + 0.35 * p_gb

        return preds

    def predict_price(self, X: pd.DataFrame, current_prices: np.ndarray) -> np.ndarray:
        """Reconstructs next-bar price P_{n+1} = current_price * (1 + predicted_return)."""
        pred_returns = self.predict_return(X)
        pred_prices = np.asarray(current_prices) * (1.0 + pred_returns)
        return pred_prices

    def _prepare_X(self, X: pd.DataFrame) -> np.ndarray:
        """Extracts and imputes required feature columns."""
        X_df = X.copy()
        for col in self.feature_columns:
            if col not in X_df.columns:
                X_df[col] = 0.0
                
        subset = X_df[self.feature_columns].fillna(0.0)
        return subset.to_numpy(dtype=float)

