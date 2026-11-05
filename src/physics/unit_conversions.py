"""
HydroPhysics AI: Physical Unit Conversions for Catchment Hydrology.

Mathematical and Dimensional Verification:
------------------------------------------
Let Q_m3_s be river volumetric discharge in cubic meters per second (m^3/s).
Let Area_km2 be the catchment drainage area in square kilometers (km^2).

Daily volume of discharge:
    V_daily [m^3/day] = Q_m3_s * 86,400 [s/day]

Catchment surface area in square meters:
    Area_m2 [m^2] = Area_km2 * 1,000,000 [m^2/km^2] = 1e6 * Area_km2

Equivalent runoff depth in meters per day:
    Depth_m_day [m/day] = V_daily / Area_m2
                        = (Q_m3_s * 86,400) / (Area_km2 * 1e6)
                        = (86,400 / 1,000,000) * (Q_m3_s / Area_km2)
                        = 0.0864 * (Q_m3_s / Area_km2)

Equivalent runoff depth in millimeters per day:
    Q_mm_day [mm/day] = Depth_m_day * 1,000 [mm/m]
                      = 0.0864 * 1,000 * (Q_m3_s / Area_km2)
                      = 86.4 * (Q_m3_s / Area_km2)

Constant:
    K = 86.4 (dimensionless scaling factor for mm/day from m^3/s and km^2)
"""

from typing import Union
import numpy as np

CONVERSION_FACTOR: float = 86.4


def discharge_m3s_to_mm_day(
    discharge_m3s: Union[float, np.ndarray],
    catchment_area_km2: float
) -> Union[float, np.ndarray]:
    """
    Convert volumetric river discharge (m^3/s) to specific catchment runoff depth (mm/day).

    Parameters
    ----------
    discharge_m3s : float or np.ndarray
        Volumetric streamflow rate at the catchment gauge in m^3/s.
    catchment_area_km2 : float
        Drainage area of the upstream catchment in km^2. Must be strictly positive.

    Returns
    -------
    float or np.ndarray
        Specific runoff depth in mm/day.

    Raises
    ------
    ValueError
        If catchment_area_km2 is <= 0 or if discharge contains negative values.
    """
    if catchment_area_km2 <= 0.0:
        raise ValueError(
            f"Catchment area must be strictly positive (>0), got {catchment_area_km2} km^2."
        )

    # Check for negative discharge
    if np.any(np.asarray(discharge_m3s) < 0.0):
        raise ValueError(
            "Discharge rate cannot be physically negative."
        )

    return (CONVERSION_FACTOR * discharge_m3s) / catchment_area_km2


def runoff_mm_day_to_m3s(
    runoff_mm_day: Union[float, np.ndarray],
    catchment_area_km2: float
) -> Union[float, np.ndarray]:
    """
    Convert specific catchment runoff depth (mm/day) to volumetric streamflow (m^3/s).

    Parameters
    ----------
    runoff_mm_day : float or np.ndarray
        Runoff depth in mm/day.
    catchment_area_km2 : float
        Drainage area of the upstream catchment in km^2.

    Returns
    -------
    float or np.ndarray
        Volumetric river streamflow in m^3/s.
    """
    if catchment_area_km2 <= 0.0:
        raise ValueError(
            f"Catchment area must be strictly positive (>0), got {catchment_area_km2} km^2."
        )

    if np.any(np.asarray(runoff_mm_day) < 0.0):
        raise ValueError(
            "Runoff depth cannot be physically negative."
        )

    return (runoff_mm_day * catchment_area_km2) / CONVERSION_FACTOR
