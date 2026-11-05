"""
HydroPhysics AI: FastAPI Scientific Web Application & REST API.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.routers import (
    catchments,
    data_explorer,
    experiments,
    health,
    models,
    physics,
)
from src.data.camels_loader import CamelsGBLoader
from src.data.downloader import generate_benchmark_camels_gb_sample
from src.training.job_runner import JobManager
from src.training.trainer import run_experiment_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("HydroPhysicsAI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler."""
    logger.info("Initializing HydroPhysics AI backend...")

    # 1. Ensure sample dataset exists
    loader = CamelsGBLoader(data_dir=settings.camels_gb_dir)
    available = loader.list_available_catchments()
    if not available:
        logger.info(f"Generating benchmark sample catchments in {settings.camels_gb_dir}...")
        generate_benchmark_camels_gb_sample(settings.camels_gb_dir)
        available = loader.list_available_catchments()
        logger.info(f"Loaded {len(available)} benchmark catchments.")

    # 2. Check if initial benchmark results exist; if not, train initial baseline for Catchment 54057
    manager = JobManager()
    has_benchmark = any(
        job.status == "completed" and str(job.config.get("gauge_id")) == "54057"
        for job in manager.jobs.values()
    )
    if not has_benchmark and available:
        logger.info("Pre-computing initial benchmark experiment for Catchment 54057 (Severn at Plynlimon)...")
        init_config = {
            "gauge_id": "54057",
            "data_proportion": 1.0,
            "noise_level": 0.0,
            "lambda_physics": 0.1,
            "lookback_window": 30,
            "epochs": 10,
            "batch_size": 32,
            "hidden_dim": 32,
            "data_dir": str(settings.camels_gb_dir),
        }
        manager.submit_job(config=init_config, task_fn=run_experiment_task)

    yield
    logger.info("Shutting down HydroPhysics AI backend.")


app = FastAPI(
    title="HydroPhysics AI: Physics-Informed ML for Rainfall-Runoff Prediction",
    description=(
        "Research-oriented API for evaluating Physics-Informed LSTM (PI-LSTM) models "
        "incorporating catchment water-balance mass conservation constraints against "
        "conventional hydrological baselines (Random Forest, LSTM) on CAMELS-GB."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(health.router)
app.include_router(catchments.router)
app.include_router(data_explorer.router)
app.include_router(models.router)
app.include_router(experiments.router)
app.include_router(physics.router)

# Mount Static Frontend
frontend_dir = settings.base_dir / "app" / "frontend"
frontend_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=True)
