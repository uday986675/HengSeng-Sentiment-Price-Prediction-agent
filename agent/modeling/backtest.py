"""Backtesting and out-of-sample validation engine comparing against naive and drift baselines."""
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from agent.modeling.predictor import SentimentPricePredictor
from agent.modeling.validation import WalkForwardTimeSeriesSplit, NaiveLastPriceBaseline, DriftBaseline, calculate_metrics
from agent.utils.logger import setup_logger

logger = setup_logger("BacktestEngine")


class BacktestResult:
    """Stores results, predictions, baseline comparisons, and trade logs."""

    def __init__(
        self,
        cv_metrics_model: Dict[str, float],
        cv_metrics_naive: Dict[str, float],
        cv_metrics_drift: Dict[str, float],
        realizations_df: pd.DataFrame,
        feature_importances: Dict[str, float],
        exchange: str
    ):
        self.cv_metrics_model = cv_metrics_model
        self.cv_metrics_naive = cv_metrics_naive
        self.cv_metrics_drift = cv_metrics_drift
        self.realizations_df = realizations_df
        self.feature_importances = feature_importances
        self.exchange = exchange

    def print_summary(self):
        """Prints a clean tabular comparison between model and baselines."""
        print("\n" + "=" * 70)
        print(f"  OUT-OF-SAMPLE WALK-FORWARD VALIDATION ({self.exchange.upper()})")
        print("=" * 70)
        header = f"{'Metric':<25} | {'Ensemble Model':<15} | {'Naive (Last)':<15} | {'Drift Baseline':<15}"
        print(header)
        print("-" * len(header))
        for m in ["MAE", "RMSE", "MAPE_pct", "Directional_Accuracy_pct"]:
            v_model = self.cv_metrics_model.get(m, 0.0)
            v_naive = self.cv_metrics_naive.get(m, 0.0)
            v_drift = self.cv_metrics_drift.get(m, 0.0)
            print(f"{m:<25} | {v_model:<15} | {v_naive:<15} | {v_drift:<15}")
        print("=" * 70)
        
        if not self.realizations_df.empty:
            print("\nREALIZED vs PREDICTED n+1 PRICE COMPARISON (Held-Out Final Period):")
            display_cols = ["date", "ticker", "current_price", "realized_n1_price", "predicted_n1_price", "error_abs", "dir_correct"]
            avail_cols = [c for c in display_cols if c in self.realizations_df.columns]
            print(self.realizations_df[avail_cols].tail(15).to_string(index=False))
            print("=" * 70 + "\n")


class BacktestEngine:
    """Executes rigorous walk-forward cross-validation and held-out backtest."""

    def __init__(self, n_splits: int = 5, test_size: int = 4, random_state: int = 42):
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.splitter = WalkForwardTimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    def run_validation(self, ml_df: pd.DataFrame, exchange: str = "hkex") -> BacktestResult:
        """
        Executes expanding-window cross validation across all tickers.
        Ensures strict temporal ordering with no lookahead.
        """
        logger.info(f"Initiating walk-forward cross-validation on {len(ml_df)} samples...")
        
        # We perform validation per ticker or pooled temporally
        # Sort strictly by timestamp
        clean_df = ml_df.dropna(subset=["target_next_close", "target_next_return"]).sort_values("timestamp").reset_index(drop=True)

        if len(clean_df) < 15:
            logger.warning("Dataset too small for multi-split walk forward. Running train-test split.")
            train_idx = np.arange(0, int(len(clean_df) * 0.7))
            test_idx = np.arange(int(len(clean_df) * 0.7), len(clean_df))
            splits = [(train_idx, test_idx)]
        else:
            splits = list(self.splitter.split(clean_df))

        all_y_true = []
        all_y_pred = []
        all_y_naive = []
        all_y_drift = []
        all_curr = []
        realization_records = []

        last_model = None

        for fold, (train_idx, test_idx) in enumerate(splits):
            train_sub = clean_df.iloc[train_idx]
            test_sub = clean_df.iloc[test_idx]

            if len(train_sub) < 5 or len(test_sub) == 0:
                continue

            # 1. Fit Model
            model = SentimentPricePredictor(model_type="ensemble", random_state=self.random_state)
            model.fit(train_sub, train_sub["target_next_return"].to_numpy())
            last_model = model

            # 2. Fit Drift
            drift = DriftBaseline()
            drift.fit(train_sub["return_1d"].to_numpy())

            # 3. Predict on Test
            curr_prices = test_sub["close"].to_numpy()
            y_true_prices = test_sub["target_next_close"].to_numpy()

            pred_prices = model.predict_price(test_sub, curr_prices)
            naive_prices = NaiveLastPriceBaseline().predict_price(curr_prices)
            drift_prices = drift.predict_price(curr_prices)

            all_y_true.extend(y_true_prices)
            all_y_pred.extend(pred_prices)
            all_y_naive.extend(naive_prices)
            all_y_drift.extend(drift_prices)
            all_curr.extend(curr_prices)

            # Record realizations for the final fold
            if fold == (len(splits) - 1):
                for idx_sub, row in enumerate(test_sub.itertuples()):
                    c_p = float(row.close)
                    r_p = float(row.target_next_close)
                    p_p = float(pred_prices[idx_sub])
                    err = abs(r_p - p_p)
                    dir_real = (r_p - c_p) > 0
                    dir_pred = (p_p - c_p) > 0
                    realization_records.append({
                        "date": str(row.timestamp)[:10],
                        "ticker": row.ticker,
                        "current_price": round(c_p, 2),
                        "realized_n1_price": round(r_p, 2),
                        "predicted_n1_price": round(p_p, 2),
                        "error_abs": round(err, 2),
                        "dir_correct": (dir_real == dir_pred)
                    })

        metrics_model = calculate_metrics(all_y_true, all_y_pred, all_curr)
        metrics_naive = calculate_metrics(all_y_true, all_y_naive, all_curr)
        metrics_drift = calculate_metrics(all_y_true, all_y_drift, all_curr)

        feature_importances = last_model.feature_importances_ if last_model else {}
        realizations_df = pd.DataFrame(realization_records)

        result = BacktestResult(
            cv_metrics_model=metrics_model,
            cv_metrics_naive=metrics_naive,
            cv_metrics_drift=metrics_drift,
            realizations_df=realizations_df,
            feature_importances=feature_importances,
            exchange=exchange
        )
        return result

