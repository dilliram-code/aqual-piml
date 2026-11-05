"""
HydroPhysics AI: Chronological Splitting, Sliding Lookback Windows, and Data Scarcity Subsetting.

Temporal Leakage Prevention:
----------------------------
Hydrological timeseries exhibit strong serial correlation and seasonal periodicity.
Random K-fold cross-validation or random shuffling introduces severe temporal data leakage.
Strict chronological splitting (Train -> Val -> Test) is strictly enforced:
- Training period: e.g. 2000-01-01 to 2009-12-31 (~65% or configured)
- Validation period: e.g. 2010-01-01 to 2011-12-31 (~15% for early stopping / tuning)
- Test evaluation period: e.g. 2012-01-01 to 2015-09-30 (~20% held-out unseen evaluation)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.scaler import HydrologicalScaler


class CatchmentDataset(Dataset):
    """
    PyTorch Dataset providing sliding lookback windows for hydrometeorological sequence models.
    Returns:
        x: Scaled input sequence tensor [lookback_window, n_features]
        y: Target discharge [forecast_horizon]
        p_unscaled: Physical precipitation over window [lookback_window]
        pet_unscaled: Physical PET over window [lookback_window]
    """

    def __init__(
        self,
        x_scaled: np.ndarray,
        y_scaled: np.ndarray,
        p_unscaled: np.ndarray,
        pet_unscaled: np.ndarray,
        y_unscaled: np.ndarray,
        dates: pd.DatetimeIndex,
        lookback_window: int = 30,
        forecast_horizon: int = 1,
    ):
        self.x_scaled = torch.tensor(x_scaled, dtype=torch.float32)
        self.y_scaled = torch.tensor(y_scaled, dtype=torch.float32)
        self.p_unscaled = torch.tensor(p_unscaled, dtype=torch.float32)
        self.pet_unscaled = torch.tensor(pet_unscaled, dtype=torch.float32)
        self.y_unscaled = torch.tensor(y_unscaled, dtype=torch.float32)
        self.dates = dates
        self.lookback = lookback_window
        self.horizon = forecast_horizon
        self.num_samples = len(x_scaled) - lookback_window - forecast_horizon + 1

    def __len__(self) -> int:
        return max(0, self.num_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx
        end = idx + self.lookback
        target_start = end
        target_end = end + self.horizon

        return {
            "x": self.x_scaled[start:end],
            "y": self.y_scaled[target_start:target_end].squeeze(-1),
            "p_unscaled": self.p_unscaled[start:end],
            "pet_unscaled": self.pet_unscaled[start:end],
            "p_target_unscaled": self.p_unscaled[target_start:target_end].squeeze(-1),
            "pet_target_unscaled": self.pet_unscaled[target_start:target_end].squeeze(-1),
            "y_target_unscaled": self.y_unscaled[target_start:target_end].squeeze(-1),
        }


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.65,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-series chronologically into Train, Validation, and Test partitions.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def chronological_date_split(
    df: pd.DataFrame,
    train_end_date: str = "2009-12-31",
    val_end_date: str = "2011-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-series by explicit calendar dates.
    """
    t_end = pd.to_datetime(train_end_date)
    v_end = pd.to_datetime(val_end_date)

    train_df = df[df.index <= t_end].copy()
    val_df = df[(df.index > t_end) & (df.index <= v_end)].copy()
    test_df = df[df.index > v_end].copy()

    return train_df, val_df, test_df


def apply_data_scarcity(
    train_df: pd.DataFrame,
    proportion: float = 1.0,
    sampling_strategy: str = "contiguous_recent",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate data-scarce hydrologic monitoring regimes:
    proportion in {1.0, 0.50, 0.25, 0.10}.

    Parameters
    ----------
    train_df : pd.DataFrame
        Full training set.
    proportion : float
        Fraction of training observations to retain (e.g. 0.25 = 25%).
    sampling_strategy : str
        'contiguous_recent' (keep the most recent proportion of the training series),
        'contiguous_early', or 'systematic_intermittent' (periodic downsampling).
    """
    if proportion >= 0.999:
        return train_df.copy()

    n = len(train_df)
    k = max(int(n * proportion), 60)  # retain at least 60 days for windowing

    if sampling_strategy == "contiguous_recent":
        # Realistic scenario: gauge was only installed in recent years
        return train_df.iloc[-k:].copy()
    elif sampling_strategy == "contiguous_early":
        return train_df.iloc[:k].copy()
    elif sampling_strategy == "systematic_intermittent":
        # Periodic observation (e.g. weekly manual gauge reading)
        step = int(np.round(1.0 / proportion))
        indices = np.arange(0, n, step)
        return train_df.iloc[indices].copy()
    else:
        raise ValueError(f"Unknown sampling strategy '{sampling_strategy}'")


def inject_measurement_noise(
    df: pd.DataFrame,
    noise_level: float = 0.0,
    target_columns: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Inject controlled Gaussian measurement noise into observations to test model robustness.
    Noise level is proportional to variable standard deviation:
        x_noisy = x + N(0, noise_level * std(x))

    Physical non-negativity is preserved by clipping at 0.0.
    """
    if noise_level <= 1e-6:
        return df.copy()

    out_df = df.copy()
    rng = np.random.default_rng(seed)

    if target_columns is None:
        target_columns = ["precipitation", "discharge_spec"]

    for col in target_columns:
        if col in out_df.columns:
            vals = out_df[col].values
            std_val = float(np.nanstd(vals))
            noise = rng.normal(0, noise_level * std_val, size=len(vals))
            noisy_vals = np.maximum(0.0, vals + noise)
            out_df[col] = np.round(noisy_vals, 3)

    return out_df


def build_feature_matrices_and_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "discharge_spec",
    lookback_window: int = 30,
    forecast_horizon: int = 1,
    scaling_method: str = "standard",
) -> Tuple[CatchmentDataset, CatchmentDataset, CatchmentDataset, HydrologicalScaler]:
    """
    End-to-end pipeline to fit scalers on training set only, scale inputs,
    and construct PyTorch CatchmentDataset objects.
    """
    if feature_cols is None:
        feature_cols = ["precipitation", "pet", "temperature"]

    # Initialize and fit scaler on training data ONLY
    scaler = HydrologicalScaler(method=scaling_method)
    scaler.fit(train_df, feature_cols + [target_col])

    # Transform
    x_train_scaled = scaler.transform(train_df[feature_cols])
    y_train_scaled = scaler.transform(train_df[[target_col]])

    x_val_scaled = scaler.transform(val_df[feature_cols])
    y_val_scaled = scaler.transform(val_df[[target_col]])

    x_test_scaled = scaler.transform(test_df[feature_cols])
    y_test_scaled = scaler.transform(test_df[[target_col]])

    # Build datasets
    train_ds = CatchmentDataset(
        x_scaled=x_train_scaled,
        y_scaled=y_train_scaled,
        p_unscaled=train_df["precipitation"].values,
        pet_unscaled=train_df["pet"].values,
        y_unscaled=train_df[target_col].values,
        dates=train_df.index,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
    )

    val_ds = CatchmentDataset(
        x_scaled=x_val_scaled,
        y_scaled=y_val_scaled,
        p_unscaled=val_df["precipitation"].values,
        pet_unscaled=val_df["pet"].values,
        y_unscaled=val_df[target_col].values,
        dates=val_df.index,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
    )

    test_ds = CatchmentDataset(
        x_scaled=x_test_scaled,
        y_scaled=y_test_scaled,
        p_unscaled=test_df["precipitation"].values,
        pet_unscaled=test_df["pet"].values,
        y_unscaled=test_df[target_col].values,
        dates=test_df.index,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
    )

    return train_ds, val_ds, test_ds, scaler

