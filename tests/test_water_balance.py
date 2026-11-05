"""
Unit Tests: Catchment Water-Balance Physics, Bounds, and Residuals.
"""

import pytest
import numpy as np

from src.physics.water_balance import CatchmentWaterBalance


def test_hand_calculated_water_balance_residual():
    """
    Hand-calculated check:
    P = 12.0 mm/day
    ET = 3.0 mm/day
    Q = 4.0 mm/day
    Delta_S = 5.0 mm/day
    Residual R = 12.0 - 3.0 - 4.0 - 5.0 = 0.0 mm/day.
    """
    res = CatchmentWaterBalance.compute_daily_residual(
        p_mm_day=12.0,
        et_mm_day=3.0,
        q_mm_day=4.0,
        delta_s_mm_day=5.0,
    )
    assert np.isclose(res, 0.0)

    # Discrepancy case:
    # Delta S observed is only 2.0 mm/day -> R = 12 - 3 - 4 - 2 = 3.0 mm/day unaccounted
    res_discrepancy = CatchmentWaterBalance.compute_daily_residual(
        p_mm_day=12.0, et_mm_day=3.0, q_mm_day=4.0, delta_s_mm_day=2.0
    )
    assert np.isclose(res_discrepancy, 3.0)


def test_validate_flux_bounds_consistent():
    """Valid fluxes must return is_physically_consistent = True."""
    p = np.array([10.0, 5.0, 0.0])
    pet = np.array([4.0, 3.0, 2.0])
    et = np.array([3.5, 2.8, 1.0])  # all <= pet and >= 0
    q = np.array([2.0, 1.5, 0.5])   # >= 0
    storage = np.array([120.0, 118.0, 115.0])

    report = CatchmentWaterBalance.validate_flux_bounds(p, pet, et, q, storage)
    assert report["is_physically_consistent"] is True
    assert report["negative_precipitation_days"] == 0
    assert report["negative_actual_et_days"] == 0
    assert report["et_exceeds_pet_days"] == 0


def test_validate_flux_bounds_violation():
    """Detect when ET exceeds PET or fluxes are negative."""
    p = np.array([10.0, -1.0])  # negative P
    pet = np.array([3.0, 2.0])
    et = np.array([4.0, 1.0])  # ET > PET on day 1
    q = np.array([-0.5, 1.0])  # negative Q
    storage = np.array([10.0, -5.0])  # negative S

    report = CatchmentWaterBalance.validate_flux_bounds(p, pet, et, q, storage)
    assert report["is_physically_consistent"] is False
    assert report["negative_precipitation_days"] == 1
    assert report["et_exceeds_pet_days"] == 1
    assert report["negative_discharge_days"] == 1
    assert report["negative_storage_days"] == 1


def test_budyko_et_bounds():
    """Analytical Budyko actual ET must satisfy: 0 <= ET <= PET and 0 <= ET <= P."""
    p = np.array([0.5, 2.0, 10.0, 50.0])
    pet = np.array([4.0, 4.0, 2.0, 1.0])

    et_budyko = CatchmentWaterBalance.estimate_budyko_actual_et(p, pet, omega=2.0)

    assert np.all(et_budyko >= 0.0)
    assert np.all(et_budyko <= pet + 1e-6)
    assert np.all(et_budyko <= p + 1e-6)


def test_cumulative_mass_balance():
    """Cumulative mass balance must satisfy S_T - S_0 = sum(P) - sum(ET + Q)."""
    p = np.array([10.0, 15.0, 5.0, 0.0])
    et = np.array([2.0, 2.5, 2.0, 1.5])
    q = np.array([3.0, 6.0, 2.0, 1.0])

    cum_p, cum_out, s = CatchmentWaterBalance.compute_cumulative_mass_balance(
        p, et, q, initial_storage_mm=50.0
    )

    total_in = np.sum(p)
    total_out = np.sum(et + q)
    storage_change = s[-1] - 50.0

    assert np.isclose(storage_change, total_in - total_out)
