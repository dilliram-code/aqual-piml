"""
Unit Tests: Standard Hydrological Evaluation Metrics (NSE, KGE, RMSE, PBIAS, PICP).
"""

import pytest
import numpy as np

from src.evaluation.metrics import (
    compute_kge,
    compute_mae,
    compute_nse,
    compute_pbias,
    compute_picp_and_mpiw,
    compute_rmse,
)


def test_nse_perfect_and_mean_predictions():
    """NSE must equal 1.0 for perfect prediction and 0.0 when predicting the observation mean."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Perfect prediction
    assert np.isclose(compute_nse(y_true, y_true), 1.0)

    # Predicting mean
    y_mean = np.full_like(y_true, np.mean(y_true))
    assert np.isclose(compute_nse(y_true, y_mean), 0.0)

    # Worse than mean
    y_bad = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert compute_nse(y_true, y_bad) < 0.0


def test_kge_components():
    """KGE must equal 1.0 for perfect prediction."""
    y_true = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    kge, r, alpha, beta = compute_kge(y_true, y_true)

    assert np.isclose(kge, 1.0)
    assert np.isclose(r, 1.0)
    assert np.isclose(alpha, 1.0)
    assert np.isclose(beta, 1.0)


def test_rmse_and_mae():
    """Verify RMSE and MAE calculation."""
    y_true = np.array([2.0, 4.0])
    y_pred = np.array([4.0, 2.0])
    # diffs = [2, -2] -> abs = [2, 2] -> MAE = 2.0
    # sq_diffs = [4, 4] -> mean = 4 -> sqrt = 2.0
    assert np.isclose(compute_mae(y_true, y_pred), 2.0)
    assert np.isclose(compute_rmse(y_true, y_pred), 2.0)


def test_pbias():
    """Verify percent bias calculation."""
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([8.0, 8.0])
    # sum(y_t - y_p) = 2 + 2 = 4; sum(y_t) = 20 -> 4 / 20 = 20%
    assert np.isclose(compute_pbias(y_true, y_pred), 20.0)


def test_picp_and_mpiw():
    """Test prediction interval coverage probability and mean width."""
    y_true = np.array([5.0, 10.0, 15.0, 20.0])
    q_lower = np.array([4.0, 8.0, 16.0, 18.0])  # sample 3 fails (15 < 16)
    q_upper = np.array([6.0, 12.0, 18.0, 22.0])

    picp, mpiw = compute_picp_and_mpiw(y_true, q_lower, q_upper)
    # 3 out of 4 inside -> 75%
    assert np.isclose(picp, 75.0)
    # widths: [2, 4, 2, 4] -> mean = 3.0
    assert np.isclose(mpiw, 3.0)
