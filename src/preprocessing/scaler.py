"""
HydroPhysics AI: Normalization and Feature Scaling without Data Leakage.

CRITICAL METHODOLOGICAL RULE:
Scalers must ALWAYS be fit on the training split only.
Validation and test sets must be transformed using the fitted training statistics.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


class HydrologicalScaler:
    """
    Normalization scaler for hydrometeorological features and target discharge.
    Supports StandardScaler and MinMaxScaler with persistent parameters for saving/loading.
    """

    def __init__(self, method: str = "standard"):
        if method not in ["standard", "minmax"]:
            raise ValueError(f"Unsupported scaling method '{method}'. Choose 'standard' or 'minmax'.")
        self.method = method
        self.feature_names: List[str] = []
        self.stats: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> "HydrologicalScaler":
        """
        Fit scaling statistics on training set columns.
        """
        self.feature_names = list(feature_cols)
        self.stats = {}

        for col in self.feature_names:
            vals = df[col].values.astype(np.float64)
            mean_val = float(np.nanmean(vals))
            std_val = float(np.nanstd(vals))
            min_val = float(np.nanmin(vals))
            max_val = float(np.nanmax(vals))

            # Protect against division by zero if variance is zero
            if std_val < 1e-6:
                std_val = 1.0
            if (max_val - min_val) < 1e-6:
                max_val = min_val + 1.0

            self.stats[col] = {
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
            }

        self.is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Transform data into scaled matrix using fitted statistics.
        If columns is provided, transform those columns; otherwise transform columns
        present in df that have fitted statistics.
        """
        if not self.is_fitted:
            raise RuntimeError("HydrologicalScaler must be fitted before transform.")

        if columns is not None:
            target_cols = [c for c in columns if c in self.stats]
        else:
            target_cols = [c for c in df.columns if c in self.stats]

        if not target_cols:
            raise ValueError(f"No valid fitted columns found to transform. Available in scaler: {list(self.stats.keys())}")

        out = np.zeros((len(df), len(target_cols)), dtype=np.float32)
        for i, col in enumerate(target_cols):
            vals = df[col].values.astype(np.float64)
            stat = self.stats[col]
            if self.method == "standard":
                scaled = (vals - stat["mean"]) / stat["std"]
            else:  # minmax
                scaled = (vals - stat["min"]) / (stat["max"] - stat["min"])
            out[:, i] = scaled.astype(np.float32)

        return out

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        return self.fit(df, feature_cols).transform(df)

    def inverse_transform_target(
        self,
        scaled_target: Union[np.ndarray, float],
        target_col: str = "discharge_spec",
    ) -> Union[np.ndarray, float]:
        """
        Convert scaled target predictions back to physical hydrological units (mm/day).
        """
        if not self.is_fitted:
            raise RuntimeError("HydrologicalScaler must be fitted before inverse transform.")
        if target_col not in self.stats:
            raise KeyError(f"Target column '{target_col}' was not in fitted features: {self.feature_names}")

        stat = self.stats[target_col]
        arr = np.asarray(scaled_target, dtype=np.float64)

        if self.method == "standard":
            physical = arr * stat["std"] + stat["mean"]
        else:  # minmax
            physical = arr * (stat["max"] - stat["min"]) + stat["min"]

        # Enforce physical non-negativity constraint
        physical = np.maximum(physical, 0.0)

        if isinstance(scaled_target, float):
            return float(physical)
        return physical

    def to_dict(self) -> dict:
        """Export state for checkpoint serialization."""
        return {
            "method": self.method,
            "feature_names": self.feature_names,
            "stats": self.stats,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HydrologicalScaler":
        """Restore scaler from state dictionary."""
        scaler = cls(method=data["method"])
        scaler.feature_names = data["feature_names"]
        scaler.stats = data["stats"]
        scaler.is_fitted = data["is_fitted"]
        return scaler
