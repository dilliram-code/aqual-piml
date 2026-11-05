"""
HydroPhysics AI: Catchment Water-Balance Physics Formulation and Diagnostics.

Physical Theory & Mass Conservation:
------------------------------------
In a lumped catchment system, conservation of water mass over a discrete time step
Delta t (here, 1 day) is governed by:

    dS/dt = P(t) - ET(t) - Q(t) - G(t)

In discrete form for day t:
    Delta S_t = S_t - S_{t-1} = P_t - ET_t - Q_t - L_t

Where:
    - P_t: Catchment areal precipitation (mm/day) [Forcing input, observed]
    - ET_t: Catchment actual evapotranspiration (mm/day) [Latent flux, constrained: 0 <= ET_t <= PET_t]
    - Q_t: Specific river discharge/runoff depth (mm/day) [Output flux, observed/predicted: Q_t >= 0]
    - S_t: Total effective catchment moisture storage (mm) [Latent state: S_t >= 0]
    - Delta S_t: Storage change over the daily interval (mm/day)
    - L_t: Unmonitored boundary leakage / deep groundwater loss (mm/day). In a topographically
           closed catchment without net inter-basin karst transfer, L_t approx 0.

Physical Bounds & Invariants:
-----------------------------
1. Non-negativity of precipitation: P_t >= 0
2. Potential ET limit: 0 <= ET_t <= PET_t
   Actual evapotranspiration is water-limited when soil moisture is low,
   and energy-limited when soil moisture is high, bounded strictly by PET.
3. Non-negativity of discharge: Q_t >= 0
4. Storage non-negativity: S_t >= 0
5. Discrete Water-Balance Residual:
       R_t = P_t - ET_t - Q_t - Delta S_t

Scientific Limitations & Explicit Assumptions:
----------------------------------------------
- Snowpack Dynamics: In cold or high-elevation catchments, solid precipitation is temporarily
  stored as snowpack and released during spring melt. In the simplified 1D single-bucket formulation,
  snow processes are reflected through temperature forcing and latent storage state S_t.
- Sub-surface karst or inter-basin groundwater exchange is assumed negligible relative to main river runoff.
- Daily discretization: Euler finite differences approximate continuous flux integration;
  exact zero daily residual does not guarantee infinite-precision differential conservation.
"""

from typing import Dict, Tuple, Union
import numpy as np


class CatchmentWaterBalance:
    """
    Catchment water-balance diagnostics, flux bounds validation,
    and mass-conservation calculations.
    """

    @staticmethod
    def compute_daily_residual(
        p_mm_day: Union[float, np.ndarray],
        et_mm_day: Union[float, np.ndarray],
        q_mm_day: Union[float, np.ndarray],
        delta_s_mm_day: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Compute daily water-balance residual:
            R_t = P_t - ET_t - Q_t - Delta S_t (mm/day)

        A perfectly conserving system has R_t = 0.
        """
        p = np.asarray(p_mm_day, dtype=np.float64)
        et = np.asarray(et_mm_day, dtype=np.float64)
        q = np.asarray(q_mm_day, dtype=np.float64)
        ds = np.asarray(delta_s_mm_day, dtype=np.float64)

        return p - et - q - ds

    @staticmethod
    def validate_flux_bounds(
        p_mm_day: np.ndarray,
        pet_mm_day: np.ndarray,
        et_mm_day: np.ndarray,
        q_mm_day: np.ndarray,
        storage_mm: np.ndarray = None,
    ) -> Dict[str, Union[bool, float, int]]:
        """
        Verify physical bounds on all hydrological fluxes and storage states.

        Returns a detailed report of physical consistency violations.
        """
        p = np.asarray(p_mm_day, dtype=np.float64)
        pet = np.asarray(pet_mm_day, dtype=np.float64)
        et = np.asarray(et_mm_day, dtype=np.float64)
        q = np.asarray(q_mm_day, dtype=np.float64)

        neg_p_count = int(np.sum(p < -1e-5))
        neg_pet_count = int(np.sum(pet < -1e-5))
        neg_et_count = int(np.sum(et < -1e-5))
        neg_q_count = int(np.sum(q < -1e-5))
        et_exceeds_pet_count = int(np.sum(et > (pet + 1e-4)))

        report = {
            "is_physically_consistent": (
                neg_p_count == 0
                and neg_pet_count == 0
                and neg_et_count == 0
                and neg_q_count == 0
                and et_exceeds_pet_count == 0
            ),
            "negative_precipitation_days": neg_p_count,
            "negative_pet_days": neg_pet_count,
            "negative_actual_et_days": neg_et_count,
            "negative_discharge_days": neg_q_count,
            "et_exceeds_pet_days": et_exceeds_pet_count,
            "max_et_over_pet_excess_mm": float(
                np.max(np.maximum(0.0, et - pet))
            ),
        }

        if storage_mm is not None:
            s = np.asarray(storage_mm, dtype=np.float64)
            neg_s_count = int(np.sum(s < -1e-5))
            report["negative_storage_days"] = neg_s_count
            if neg_s_count > 0:
                report["is_physically_consistent"] = False

        return report

    @staticmethod
    def compute_cumulative_mass_balance(
        p_mm_day: np.ndarray,
        et_mm_day: np.ndarray,
        q_mm_day: np.ndarray,
        initial_storage_mm: float = 100.0,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute cumulative inflows, outflows, and implied storage sequence.

        Parameters
        ----------
        p_mm_day : np.ndarray
            Daily precipitation (mm/day)
        et_mm_day : np.ndarray
            Daily actual evapotranspiration (mm/day)
        q_mm_day : np.ndarray
            Daily discharge depth (mm/day)
        initial_storage_mm : float
            Assumed baseline moisture storage at t=0 (mm)

        Returns
        -------
        cumulative_inflow : np.ndarray
            Cumulative precipitation sum (mm)
        cumulative_outflow : np.ndarray
            Cumulative ET + Q sum (mm)
        implied_storage : np.ndarray
            Storage trajectory S_t = S_0 + cumsum(P - ET - Q)
        final_residual : float
            Overall mass conservation closure discrepancy (mm)
        """
        p = np.asarray(p_mm_day, dtype=np.float64)
        et = np.asarray(et_mm_day, dtype=np.float64)
        q = np.asarray(q_mm_day, dtype=np.float64)

        cum_p = np.cumsum(p)
        cum_out = np.cumsum(et + q)
        delta_s_cum = cum_p - cum_out
        implied_storage = initial_storage_mm + delta_s_cum

        final_closure = float(cum_p[-1] - cum_out[-1] - (implied_storage[-1] - initial_storage_mm))
        return cum_p, cum_out, implied_storage

    @staticmethod
    def estimate_budyko_actual_et(
        p_mm_day: np.ndarray,
        pet_mm_day: np.ndarray,
        omega: float = 2.0,
    ) -> np.ndarray:
        """
        Calculate reference theoretical actual ET using Zhang/Fu analytical Budyko formulation.
        Useful for benchmark physical consistency comparison:
            ET / P = 1 + (PET/P) - [1 + (PET/P)^omega]^(1/omega)

        Guarantees:
            0 <= ET <= PET and 0 <= ET <= P
        """
        p = np.asarray(p_mm_day, dtype=np.float64)
        pet = np.asarray(pet_mm_day, dtype=np.float64)

        # Avoid division by zero
        safe_p = np.maximum(p, 1e-4)
        phi = pet / safe_p  # aridity index

        # Fu equation
        fu_ratio = 1.0 + phi - np.power(1.0 + np.power(phi, omega), 1.0 / omega)
        fu_ratio = np.clip(fu_ratio, 0.0, 1.0)

        et_budyko = fu_ratio * p
        # Enforce strict upper bound by PET
        et_budyko = np.minimum(et_budyko, pet)
        et_budyko = np.maximum(et_budyko, 0.0)
        return et_budyko
