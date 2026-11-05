"""
HydroPhysics AI: Asynchronous Training Job Runner and Experiment Tracker.

Thread-Safe Background Execution:
---------------------------------
Long-running neural network training and hyperparameter ablations must NEVER block
FastAPI request worker threads. This module provides a thread-safe JobManager that
spawns worker threads, tracks real-time progress (epoch losses, completion percentage),
stores experiment metadata, and saves results to disk.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional
import uuid

import numpy as np


class TrainingJob:
    """Represents the live state of a single background training experiment."""

    def __init__(self, job_id: str, config: Dict[str, Any]):
        self.job_id = job_id
        self.config = config
        self.status: str = "queued"  # queued, running, completed, failed
        self.progress_pct: int = 0
        self.current_epoch: int = 0
        self.total_epochs: int = config.get("epochs", 20)
        self.current_train_loss: Optional[float] = None
        self.current_val_loss: Optional[float] = None
        self.status_message: str = "Job initialized in queue."
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.error: Optional[str] = None
        self.results: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress_pct": self.progress_pct,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_train_loss": self.current_train_loss,
            "current_val_loss": self.current_val_loss,
            "status_message": self.status_message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "config": self.config,
            "has_results": self.results is not None,
        }


class JobManager:
    """Thread-safe background job registry and executor."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_manager()
            return cls._instance

    def _init_manager(self):
        self.jobs: Dict[str, TrainingJob] = {}
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="HydroJob")
        self.storage_dir = Path("experiments/results")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_saved_experiments()

    def _load_saved_experiments(self):
        """Restore previous experiment runs from JSON artifacts on startup."""
        for f in self.storage_dir.glob("exp_*.json"):
            try:
                with open(f, "r") as fp:
                    data = json.load(fp)
                    job_id = data.get("job_id", f.stem)
                    job = TrainingJob(job_id=job_id, config=data.get("config", {}))
                    job.status = data.get("status", "completed")
                    job.progress_pct = 100
                    job.created_at = data.get("created_at", "")
                    job.completed_at = data.get("completed_at", "")
                    job.results = data.get("results")
                    self.jobs[job_id] = job
            except Exception:
                pass

    def submit_job(self, config: Dict[str, Any], task_fn: Callable) -> str:
        """Create and submit a training job to the background thread pool."""
        job_id = f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job = TrainingJob(job_id=job_id, config=config)
        self.jobs[job_id] = job

        self.executor.submit(self._run_job_wrapper, job, task_fn)
        return job_id

    def _run_job_wrapper(self, job: TrainingJob, task_fn: Callable):
        job.status = "running"
        job.started_at = datetime.now(timezone.utc).isoformat()
        job.status_message = "Training started in background..."

        def progress_cb(current_ep, total_ep, train_loss, val_loss):
            job.current_epoch = current_ep
            job.total_epochs = total_ep
            job.current_train_loss = round(float(train_loss), 4) if train_loss else None
            job.current_val_loss = round(float(val_loss), 4) if val_loss else None
            job.progress_pct = int((current_ep / max(1, total_ep)) * 100)
            job.status_message = f"Epoch {current_ep}/{total_ep} - Loss: {job.current_train_loss}"
            job.history.append({
                "epoch": current_ep,
                "train_loss": job.current_train_loss,
                "val_loss": job.current_val_loss,
            })

        try:
            results = task_fn(job.config, progress_cb)
            job.status = "completed"
            job.progress_pct = 100
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.status_message = "Experiment completed successfully."
            job.results = results

            # Persist experiment artifact to disk
            exp_file = self.storage_dir / f"{job.job_id}.json"
            with open(exp_file, "w") as fp:
                # Custom JSON encoder for numpy types
                def default_converter(o):
                    if isinstance(o, (np.floating, np.integer)):
                        return float(o)
                    elif isinstance(o, np.ndarray):
                        return o.tolist()
                    return str(o)

                json.dump({
                    "job_id": job.job_id,
                    "status": job.status,
                    "config": job.config,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "completed_at": job.completed_at,
                    "history": job.history,
                    "results": results,
                }, fp, default=default_converter, indent=2)

        except Exception as e:
            job.status = "failed"
            job.completed_at = datetime.utcnow().isoformat()
            job.error = str(e)
            job.status_message = f"Training failed: {str(e)}"
            traceback.print_exc()

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [job.to_dict() for job in sorted(
            self.jobs.values(), key=lambda j: j.created_at, reverse=True
        )]
