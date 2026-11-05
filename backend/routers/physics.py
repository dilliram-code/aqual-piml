"""
HydroPhysics AI: Physics Diagnostics, Mass Conservation, and Equations Router.
"""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException
import numpy as np

from backend.config import settings
from backend.schemas.api_schemas import PhysicsDiagnosticsResponse
from src.data.camels_loader import CamelsGBLoader
from src.physics.water_balance import CatchmentWaterBalance
from src.training.job_runner import JobManager

router = APIRouter(prefix="/api/physics", tags=["Physics Diagnostics"])


@router.get("/equations", response_model=PhysicsDiagnosticsResponse)
def get_physics_equations():
    """Retrieve theoretical equations, units, assumptions, and physical invariants."""
    return PhysicsDiagnosticsResponse(
        equation="Delta S_t = S_t - S_{t-1} = P_t - ET_t - Q_t - L_t",
        variables={
            "P_t": "Areal Precipitation (mm/day) [Meteorological Forcing, Observed]",
            "ET_t": "Actual Evapotranspiration (mm/day) [Latent Flux, 0 <= ET_t <= PET_t]",
            "PET_t": "Potential Evapotranspiration (mm/day) [Upper Bound, Penman-Monteith]",
            "Q_t": "Specific Catchment Discharge (mm/day) [Observed/Predicted, Q_t >= 0]",
            "S_t": "Catchment Moisture Storage (mm) [Latent State, S_t >= 0]",
            "Delta S_t": "Daily Storage Increment (mm/day)",
            "L_t": "Boundary Leakage / Inter-basin transfer (assumed approx 0)",
            "R_t": "Discrete Daily Water-Balance Discrepancy Residual (mm/day)",
        },
        units={
            "Precipitation": "mm/day",
            "Actual ET": "mm/day",
            "Potential ET": "mm/day",
            "Specific Discharge": "mm/day",
            "Volumetric River Streamflow": "m^3/s (converted via Q_mm_day = 86.4 * Q_m3_s / Area_km2)",
            "Moisture Storage": "mm",
            "Catchment Drainage Area": "km^2",
        },
        assumptions=[
            "Topographically closed catchment boundary with negligible unmonitored karst leakage.",
            "Potential ET represents an upper thermodynamic limit; actual ET <= PET strictly.",
            "Discharge is non-negative: Q >= 0 enforced via Softplus non-linear activation.",
            "Storage is strictly non-negative: S >= 0.",
            "Precipitation is liquid rain or melt; snowpack delay is captured via temperature and latent storage states.",
        ],
        limitations=[
            "Daily finite-difference discretization is an approximation of continuous dS/dt differential conservation.",
            "A discrete residual R_t approx 0 at sampled timesteps regularizes the network but does not mathematically guarantee infinite-precision differential conservation across all unobserved intermediate states.",
            "Subsurface deep groundwater aquifer residence times (multi-year baseflow) are parameterized lumpedly.",
        ],
        summary_metrics={
            "conversion_factor": 86.4,
            "latent_et_constraint": "ET_pred = Sigmoid(h) * PET",
            "storage_constraint": "S_pred = Softplus(h) >= 0",
            "streamflow_constraint": "Q_pred = Softplus(h) >= 0",
        },
    )


@router.get("/water-balance/{gauge_id}")
def get_catchment_water_balance(gauge_id: str):
    """Retrieve actual ET, storage trajectory, and mass balance closure for a catchment."""
    # Check if a completed experiment exists for this catchment
    manager = JobManager()
    matching_job = None
    for job in manager.jobs.values():
        if (
            job.status == "completed"
            and job.results
            and str(job.config.get("gauge_id")) == str(gauge_id)
        ):
            matching_job = job
            break

    if matching_job and matching_job.results:
        preds = matching_job.results["predictions"]
        dates = preds["dates"]
        p = np.array(preds["precipitation_mm"])
        pet = np.array(preds["pet_mm"])
        q = np.array(preds.get("pi_lstm_pred_mm", preds["observed_mm"]))
        et = np.array(preds.get("et_pred_mm", CatchmentWaterBalance.estimate_budyko_actual_et(p, pet)))
        storage = np.array(preds.get("storage_mm", np.zeros_like(p)))
        delta_s = np.array(preds.get("delta_s_mm", np.zeros_like(p)))
    else:
        loader = CamelsGBLoader(data_dir=settings.camels_gb_dir)
        if str(gauge_id) not in loader.list_available_catchments():
            raise HTTPException(status_code=404, detail=f"Catchment {gauge_id} not found.")
        df = loader.load_catchment_timeseries(gauge_id)
        # Take the last 730 days (2 years) for clear visualization
        df_sub = df.iloc[-730:]
        dates = [str(d.date()) for d in df_sub.index]
        p = df_sub["precipitation"].values
        pet = df_sub["pet"].values
        q = df_sub["discharge_spec"].values
        et = CatchmentWaterBalance.estimate_budyko_actual_et(p, pet)
        cum_p, cum_out, storage = CatchmentWaterBalance.compute_cumulative_mass_balance(p, et, q)
        delta_s = np.diff(storage, prepend=storage[0])

    # Residuals
    residual = p - et - q - delta_s
    cum_in = np.cumsum(p)
    cum_out = np.cumsum(et + q)

    return {
        "gauge_id": gauge_id,
        "count": len(dates),
        "dates": dates,
        "precipitation_mm": [round(float(v), 2) for v in p],
        "pet_mm": [round(float(v), 2) for v in pet],
        "actual_et_mm": [round(float(v), 2) for v in et],
        "discharge_mm": [round(float(v), 3) for v in q],
        "storage_mm": [round(float(v), 2) for v in storage],
        "delta_s_mm": [round(float(v), 3) for v in delta_s],
        "daily_residual_mm": [round(float(v), 3) for v in residual],
        "cumulative_inflow_mm": [round(float(v), 1) for v in cum_in],
        "cumulative_outflow_mm": [round(float(v), 1) for v in cum_out],
        "mean_abs_residual_mm": round(float(np.mean(np.abs(residual))), 4),
        "total_mass_balance_closure_mm": round(float(cum_in[-1] - cum_out[-1] - (storage[-1] - storage[0])), 2),
    }

