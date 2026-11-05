"""
HydroPhysics AI: Model Inference and Comparison Router.
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Query
import numpy as np

from backend.config import settings
from backend.schemas.api_schemas import PredictRequest
from src.data.camels_loader import CamelsGBLoader
from src.evaluation.benchmark import BenchmarkRunner
from src.training.job_runner import JobManager

router = APIRouter(prefix="/api/models", tags=["Models & Inference"])


@router.post("/predict")
def run_model_inference(req: PredictRequest):
    """Run model inference for a selected catchment and return predictions with uncertainty."""
    loader = CamelsGBLoader(data_dir=settings.camels_gb_dir)
    available = loader.list_available_catchments()
    if str(req.gauge_id) not in available:
        raise HTTPException(status_code=404, detail=f"Catchment {req.gauge_id} not found.")

    # Check if there is already a completed experiment for this catchment in JobManager
    manager = JobManager()
    matching_job = None
    for job in manager.jobs.values():
        if (
            job.status == "completed"
            and job.results
            and str(job.config.get("gauge_id")) == str(req.gauge_id)
        ):
            matching_job = job
            break

    if matching_job and matching_job.results:
        preds = matching_job.results["predictions"]
        model_key = req.model_type.lower()
        if model_key not in ["random_forest", "conventional_lstm", "physics_informed_lstm"]:
            model_key = "physics_informed_lstm"

        key_prefix = "pi_lstm" if model_key == "physics_informed_lstm" else ("lstm" if model_key == "conventional_lstm" else "rf")

        return {
            "gauge_id": req.gauge_id,
            "model_type": model_key,
            "dates": preds["dates"],
            "observed_mm": preds["observed_mm"],
            "predicted_mm": preds.get(f"{key_prefix}_pred_mm", []),
            "lower_bound_mm": preds.get(f"{key_prefix}_lower_mm", []),
            "upper_bound_mm": preds.get(f"{key_prefix}_upper_mm", []),
            "precipitation_mm": preds["precipitation_mm"],
            "pet_mm": preds["pet_mm"],
            "is_from_saved_experiment": True,
            "job_id": matching_job.job_id,
        }

    # If no cached run exists, run a fast CPU benchmark runner for immediate results
    df = loader.load_catchment_timeseries(req.gauge_id)
    meta = loader.get_catchment_metadata(req.gauge_id)
    runner = BenchmarkRunner(
        lookback_window=req.lookback_window,
        forecast_horizon=1,
        epochs=8,
        batch_size=32,
        hidden_dim=32,
        lambda_physics=0.1,
    )
    res = runner.run_catchment_benchmark(df=df, catchment_area_km2=float(meta.get("area", 100.0)))
    preds = res["predictions"]
    model_key = req.model_type.lower()
    key_prefix = "pi_lstm" if model_key == "physics_informed_lstm" else ("lstm" if model_key == "conventional_lstm" else "rf")

    return {
        "gauge_id": req.gauge_id,
        "model_type": model_key,
        "dates": preds["dates"],
        "observed_mm": preds["observed_mm"],
        "predicted_mm": preds.get(f"{key_prefix}_pred_mm", []),
        "lower_bound_mm": preds.get(f"{key_prefix}_lower_mm", []),
        "upper_bound_mm": preds.get(f"{key_prefix}_upper_mm", []),
        "precipitation_mm": preds["precipitation_mm"],
        "pet_mm": preds["pet_mm"],
        "is_from_saved_experiment": False,
    }


@router.get("/compare/{gauge_id}")
def get_model_comparison(gauge_id: str):
    """Retrieve side-by-side performance metrics across all models for a catchment."""
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
        return {
            "gauge_id": gauge_id,
            "source": f"Experiment {matching_job.job_id}",
            "models": matching_job.results["models"],
            "metadata": matching_job.results["metadata"],
        }

    # Fallback to computing comparison
    loader = CamelsGBLoader(data_dir=settings.camels_gb_dir)
    if str(gauge_id) not in loader.list_available_catchments():
        raise HTTPException(status_code=404, detail=f"Catchment {gauge_id} not found.")

    df = loader.load_catchment_timeseries(gauge_id)
    meta = loader.get_catchment_metadata(gauge_id)
    runner = BenchmarkRunner(
        lookback_window=30,
        forecast_horizon=1,
        epochs=8,
        batch_size=32,
        hidden_dim=32,
        lambda_physics=0.1,
    )
    res = runner.run_catchment_benchmark(df=df, catchment_area_km2=float(meta.get("area", 100.0)))
    return {
        "gauge_id": gauge_id,
        "source": "On-demand execution",
        "models": res["models"],
        "metadata": res["metadata"],
    }

