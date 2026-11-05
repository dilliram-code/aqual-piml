"""
Unit Tests: Chronological Data Splits, Normalization Scalers, and Data Scarcity.
"""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing.scaler import HydrologicalScaler
from src.preprocessing.sequence_builder import (
    apply_data_scarcity,
    build_feature_matrices_and_datasets,
    chronological_split,
    inject_measurement_noise,
)


@pytest.fixture
def sample_timeseries_df():
    dates = pd.date_range("2000-01-01", periods=1000, freq="D")
    df = pd.DataFrame(
        {
            "precipitation": np.random.uniform(0, 20, size=1000),
            "pet": np.random.uniform(0.5, 5, size=1000),
            "temperature": np.random.uniform(0, 25, size=1000),
            "discharge_spec": np.random.uniform(0.1, 15, size=1000),
        },
        index=dates,
    )
    return df


def test_chronological_split_no_leakage(sample_timeseries_df):
    """Chronological split must preserve temporal ordering with zero overlap."""
    train_df, val_df, test_df = chronological_split(sample_timeseries_df, 0.60, 0.20)

    assert len(train_df) == 600
    assert len(val_df) == 200
    assert len(test_df) == 200

    # Ensure dates do not overlap
    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()


def test_scaler_no_leakage(sample_timeseries_df):
    """
    Scaler fitted on training split only must not use validation or test data.
    Inverse transform must correctly recover original physical discharge.
    """
    train_df, val_df, test_df = chronological_split(sample_timeseries_df, 0.70, 0.15)
    features = ["precipitation", "pet", "temperature", "discharge_spec"]

    scaler = HydrologicalScaler(method="standard")
    scaler.fit(train_df, features)

    # Scaler mean must match train mean, NOT overall mean
    train_mean_q = float(train_df["discharge_spec"].mean())
    assert np.isclose(scaler.stats["discharge_spec"]["mean"], train_mean_q, atol=1e-5)

    # Test inverse transformation
    original_q = test_df["discharge_spec"].values[:10]
    scaled_q = scaler.transform(test_df[features])[:10, 3]
    recovered_q = scaler.inverse_transform_target(scaled_q, "discharge_spec")

    assert np.allclose(original_q, recovered_q, atol=1e-4)


def test_apply_data_scarcity(sample_timeseries_df):
    """Verify data scarcity reductions (50%, 25%, 10%)."""
    train_df, _, _ = chronological_split(sample_timeseries_df, 0.70, 0.15)
    n_train = len(train_df)

    scarcity_50 = apply_data_scarcity(train_df, proportion=0.50)
    assert len(scarcity_50) == int(n_train * 0.50)

    scarcity_25 = apply_data_scarcity(train_df, proportion=0.25)
    assert len(scarcity_25) == int(n_train * 0.25)

    scarcity_10 = apply_data_scarcity(train_df, proportion=0.10)
    assert len(scarcity_10) == max(int(n_train * 0.10), 60)


def test_inject_measurement_noise(sample_timeseries_df):
    """Noise injection must alter values while maintaining physical non-negativity."""
    noisy_df = inject_measurement_noise(sample_timeseries_df, noise_level=0.15, seed=42)

    assert not np.allclose(noisy_df["precipitation"].values, sample_timeseries_df["precipitation"].values)
    # Non-negativity constraint preserved
    assert np.all(noisy_df["precipitation"].values >= 0.0)
    assert np.all(noisy_df["discharge_spec"].values >= 0.0)


def test_dataset_sliding_windows(sample_timeseries_df):
    """Verify PyTorch dataset shapes for sliding lookback windows."""
    train_df, val_df, test_df = chronological_split(sample_timeseries_df, 0.60, 0.20)
    lookback = 30
    horizon = 1

    train_ds, _, _, _ = build_feature_matrices_and_datasets(
        train_df, val_df, test_df, lookback_window=lookback, forecast_horizon=horizon
    )

    expected_samples = len(train_df) - lookback - horizon + 1
    assert len(train_ds) == expected_samples

    item = train_ds[0]
    assert item["x"].shape == (lookback, 3)
    assert item["y"].ndim == 0 or item["y"].shape == ()
    assert item["p_unscaled"].shape == (lookback,)
