"""
HydroPhysics AI: Random Forest / Gradient Boosting Hydrological Baseline.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.base import BaseHydrologicalModel
from src.preprocessing.scaler import HydrologicalScaler


class RandomForestHydrologyModel(BaseHydrologicalModel):
    """
    Random Forest baseline regressor using lagged meteorological forcing features.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 5,
        n_lags: int = 7,
        random_state: int = 42,
    ):
        super().__init__(name="Random Forest Baseline", model_type="random_forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_lags = n_lags
        self.random_state = random_state

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.feature_cols: List[str] = ["precipitation", "pet", "temperature"]
        self.lag_feature_names: List[str] = []
        self.config = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "n_lags": self.n_lags,
            "random_state": self.random_state,
        }

    def _create_lag_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create autoregressive and meteorological lagged input vectors."""
        feature_data = []
        feature_names = []

        for lag in range(self.n_lags + 1):
            for col in self.feature_cols:
                if col in df.columns:
                    col_lagged = df[col].shift(lag)
                    feature_data.append(col_lagged)
                    feature_names.append(f"{col}_lag{lag}")

        lag_df = pd.concat(feature_data, axis=1)
        lag_df.columns = feature_names
        self.lag_feature_names = feature_names

        # Target: discharge_spec
        target = df["discharge_spec"].values if "discharge_spec" in df.columns else np.zeros(len(df))

        # Drop NaNs from lagging
        valid_idx = ~lag_df.isna().any(axis=1)
        X = lag_df[valid_idx].values
        y = target[valid_idx]

        return X, y

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Fit Random Forest on lagged meteorological features."""
        X_train, y_train = self._create_lag_features(train_df)

        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_preds = np.maximum(0.0, self.model.predict(X_train))
        train_rmse = float(np.sqrt(np.mean((train_preds - y_train) ** 2)))

        val_rmse = None
        if val_df is not None:
            X_val, y_val = self._create_lag_features(val_df)
            val_preds = np.maximum(0.0, self.model.predict(X_val))
            val_rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))

        return {
            "epochs": [1],
            "train_loss": [train_rmse],
            "val_loss": [val_rmse] if val_rmse is not None else [],
            "feature_importances": dict(
                zip(self.lag_feature_names, self.model.feature_importances_.tolist())
            ),
        }

    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict non-negative streamflow depth (mm/day)."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict.")

        X, _ = self._create_lag_features(df)
        preds = self.model.predict(X)
        # Pad with zeros or first prediction for the initial lag days
        n_dropped = self.n_lags
        full_preds = np.zeros(len(df), dtype=np.float64)
        full_preds[:n_dropped] = preds[0] if len(preds) > 0 else 0.0
        full_preds[n_dropped:] = np.maximum(0.0, preds)
        return full_preds

    def predict_with_uncertainty(
        self, df: pd.DataFrame, n_samples: int = 50, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate tree-ensemble variance and prediction intervals across individual trees.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict.")

        X, _ = self._create_lag_features(df)
        # Get predictions from every individual estimator tree
        tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])  # [n_trees, n_samples]
        tree_preds = np.maximum(0.0, tree_preds)

        p5 = np.percentile(tree_preds, 5, axis=0)
        p50 = np.percentile(tree_preds, 50, axis=0)
        p95 = np.percentile(tree_preds, 95, axis=0)

        n_dropped = self.n_lags
        full_p50 = np.zeros(len(df))
        full_p5 = np.zeros(len(df))
        full_p95 = np.zeros(len(df))

        full_p50[:n_dropped] = p50[0] if len(p50) > 0 else 0.0
        full_p50[n_dropped:] = p50

        full_p5[:n_dropped] = p5[0] if len(p5) > 0 else 0.0
        full_p5[n_dropped:] = p5

        full_p95[:n_dropped] = p95[0] if len(p95) > 0 else 0.0
        full_p95[n_dropped:] = p95

        return full_p50, full_p5, full_p95

    def save(self, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "lag_feature_names": self.lag_feature_names,
            "config": self.config,
            "is_trained": self.is_trained,
        }
        joblib.dump(state, filepath)

    def load(self, filepath: Union[str, Path]) -> None:
        state = joblib.load(filepath)
        self.model = state["model"]
        self.feature_cols = state["feature_cols"]
        self.lag_feature_names = state["lag_feature_names"]
        self.config = state["config"]
        self.is_trained = state["is_trained"]
