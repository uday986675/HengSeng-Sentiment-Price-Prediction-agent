"""Rigorous temporal cross-validation splitters and benchmark baselines."""
from typing import Generator, List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class WalkForwardTimeSeriesSplit:
    """
    Expanding-window walk-forward cross-validation splitter.
    Ensures strict temporal ordering with zero lookahead bias and no shuffling.
    """

    def __init__(self, n_splits: int = 5, min_train_size: int = 15, test_size: int = 5):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size

    def split(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yields (train_indices, test_indices) tuples."""
        n_samples = len(df)
        total_test_needed = self.n_splits * self.test_size
        
        if n_samples < (self.min_train_size + self.test_size):
            # Fallback for very short series
            train_end = max(3, int(n_samples * 0.7))
            yield np.arange(0, train_end), np.arange(train_end, n_samples)
            return

        step = max(1, (n_samples - self.min_train_size - self.test_size) // max(1, self.n_splits - 1))
        
        for i in range(self.n_splits):
            train_end = self.min_train_size + (i * step)
            test_start = train_end
            test_end = min(n_samples, test_start + self.test_size)
            
            if test_start >= n_samples:
                break
                
            yield np.arange(0, train_end), np.arange(test_start, test_end)


class NaiveLastPriceBaseline:
    """Predicts next-bar price is exactly the current bar price (Zero Return)."""

    def predict_price(self, current_prices: np.ndarray) -> np.ndarray:
        return np.array(current_prices)

    def predict_return(self, n_samples: int) -> np.ndarray:
        return np.zeros(n_samples)


class DriftBaseline:
    """Predicts next-bar price following historical empirical drift."""

    def __init__(self):
        self.drift_ = 0.0

    def fit(self, historical_returns: np.ndarray):
        valid = historical_returns[~np.isnan(historical_returns)]
        self.drift_ = float(np.mean(valid)) if len(valid) > 0 else 0.0
        return self

    def predict_price(self, current_prices: np.ndarray) -> np.ndarray:
        return np.array(current_prices) * (1.0 + self.drift_)

    def predict_return(self, n_samples: int) -> np.ndarray:
        return np.full(n_samples, self.drift_)


def calculate_metrics(y_true_price: np.ndarray, y_pred_price: np.ndarray, current_price: np.ndarray) -> Dict[str, float]:
    """Computes MAE, RMSE, MAPE, and Directional Accuracy."""
    y_true_price = np.asarray(y_true_price)
    y_pred_price = np.asarray(y_pred_price)
    current_price = np.asarray(current_price)

    mae = float(mean_absolute_error(y_true_price, y_pred_price))
    rmse = float(np.sqrt(mean_squared_error(y_true_price, y_pred_price)))
    
    # MAPE
    valid_mask = y_true_price > 0
    if np.sum(valid_mask) > 0:
        mape = float(np.mean(np.abs((y_true_price[valid_mask] - y_pred_price[valid_mask]) / y_true_price[valid_mask])) * 100.0)
    else:
        mape = 0.0

    # Directional Accuracy
    realized_return = (y_true_price - current_price) / (current_price + 1e-8)
    predicted_return = (y_pred_price - current_price) / (current_price + 1e-8)
    
    correct_direction = (np.sign(realized_return) == np.sign(predicted_return)) | (np.abs(realized_return) < 1e-5)
    directional_accuracy = float(np.mean(correct_direction) * 100.0)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE_pct": round(mape, 2),
        "Directional_Accuracy_pct": round(directional_accuracy, 2)
    }

