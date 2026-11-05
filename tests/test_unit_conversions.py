"""
Unit Tests: Hydrological Unit Conversions and Dimensional Verification.
"""

import pytest
import numpy as np

from src.physics.unit_conversions import (
    CONVERSION_FACTOR,
    discharge_m3s_to_mm_day,
    runoff_mm_day_to_m3s,
)


def test_conversion_constant():
    """Verify that the constant 86.4 is derived from exact dimensional analysis."""
    # (86400 s/day * 1000 mm/m) / (1,000,000 m^2/km^2) = 86.4
    expected = (86400.0 * 1000.0) / 1e6
    assert np.isclose(CONVERSION_FACTOR, expected)


def test_hand_calculated_conversion():
    """
    Hand-calculated benchmark:
    If catchment area = 86.4 km^2 and streamflow Q = 10.0 m^3/s:
    Q_mm_day = 86.4 * 10.0 / 86.4 = 10.0 mm/day.
    """
    area = 86.4
    q_m3s = 10.0
    q_mm = discharge_m3s_to_mm_day(q_m3s, catchment_area_km2=area)
    assert np.isclose(q_mm, 10.0)


def test_roundtrip_conversion():
    """Converting from m^3/s to mm/day and back must yield original value within float precision."""
    area = 245.5
    original_m3s = np.array([0.0, 1.25, 45.8, 120.3, 0.05])
    mm_day = discharge_m3s_to_mm_day(original_m3s, catchment_area_km2=area)
    recovered_m3s = runoff_mm_day_to_m3s(mm_day, catchment_area_km2=area)
    assert np.allclose(original_m3s, recovered_m3s, atol=1e-7)


def test_invalid_area_raises():
    """Catchment area <= 0 must raise ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        discharge_m3s_to_mm_day(10.0, catchment_area_km2=0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        discharge_m3s_to_mm_day(10.0, catchment_area_km2=-50.0)

    with pytest.raises(ValueError, match="strictly positive"):
        runoff_mm_day_to_m3s(5.0, catchment_area_km2=-10.0)


def test_negative_discharge_raises():
    """Streamflow cannot be physically negative."""
    with pytest.raises(ValueError, match="cannot be physically negative"):
        discharge_m3s_to_mm_day(-2.5, catchment_area_km2=100.0)

    with pytest.raises(ValueError, match="cannot be physically negative"):
        runoff_mm_day_to_m3s(np.array([1.0, -0.5, 3.0]), catchment_area_km2=100.0)

