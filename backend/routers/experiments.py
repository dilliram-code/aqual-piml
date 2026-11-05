"""
HydroPhysics AI: Asynchronous Experiment Execution, Tracking, and Export Router.
"""

import io
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import pandas as pd

from backend.config import settings
from backend.schemas.api_schemas import JobStatusResponse, LaunchExperimentRequest
from src.training.job_runner import JobManager
from src.training.trainer import run_experiment_task

router = APIRouter(prefix="/api/experiments", tags=["Experiments"])


@router.post("/launch", response_model=Dict[str, str])
def launch_training_experiment(req: LaunchExperimentRequest):
    """
    Launch asynchronous model training in a background thread without blocking the API.
    """
    manager = JobManager()
    config = req.model_dump()
    config["data_dir"] = str(settings.camels_gb_dir)

    job_id = manager.submit_job(config=config, task_fn=run_experiment_task)
    return {
        "job_id": job_id,
        "message": f"Training experiment {job_id} submitted and executing in background.",
    }


@router.get("", response_model=List[JobStatusResponse])
def list_experiments():
    """List all historical and active training experiments."""
    manager = JobManager()
    return manager.list_jobs()


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_experiment_status(job_id: str):
    """Poll live progress (epoch, loss, percentage) of an ongoing or completed job."""
    manager = JobManager()
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Experiment '{job_id}' not found.")
    return job.to_dict()


@router.get("/{job_id}/results")
def get_experiment_results(job_id: str):
    """Retrieve full evaluation metrics, prediction timeseries, and water-balance fluxes."""
    manager = JobManager()
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Experiment '{job_id}' not found.")

    if job.status != "completed":
        return {
            "status": job.status,
            "message": f"Experiment is currently {job.status}. Results not yet available.",
            "progress_pct": job.progress_pct,
            "current_epoch": job.current_epoch,
            "total_epochs": job.total_epochs,
        }

    return {
        "job_id": job.job_id,
        "status": job.status,
        "config": job.config,
        "results": job.results,
        "history": job.history,
    }


@router.get("/{job_id}/export/csv")
def export_experiment_csv(job_id: str):
    """Download experiment predictions and metrics as a CSV file."""
    manager = JobManager()
    job = manager.get_job(job_id)
    if not job or not job.results:
        raise HTTPException(status_code=404, detail=f"Completed experiment '{job_id}' not found.")

    preds = job.results.get("predictions", {})
    if not preds or "dates" not in preds:
        raise HTTPException(status_code=400, detail="No prediction timeseries available to export.")

    df = pd.DataFrame(preds)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")

    filename = f"hydrophysics_predictions_{job_id}.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
