"""
HydroPhysics AI: System Health and Readiness Diagnostics Router.
"""

from pathlib import Path
from fastapi import APIRouter
from backend.config import settings
from backend.schemas.api_schemas import HealthResponse
from src.data.camels_loader import CamelsGBLoader

router = APIRouter(prefix="/api", tags=["Health & Status"])


@router.get("/health", response_model=HealthResponse)
def get_system_health():
    loader = CamelsGBLoader(data_dir=settings.camels_gb_dir)
    available = loader.list_available_catchments()
    data_exists = len(available) > 0

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        app_name=settings.app_name,
        dataset_status="ready" if data_exists else "no_data_found",
        available_catchments=len(available),
        models_ready=True,
    )


@router.get("/status")
def get_detailed_status():
    loader = CamelsGBLoader(data_dir=settings.camels_gb_dir)
    catchments = loader.list_available_catchments()
    return {
        "status": "operational",
        "camels_gb_directory": str(settings.camels_gb_dir),
        "dataset_ready": len(catchments) > 0,
        "catchment_ids": catchments,
        "pytorch_device": "cpu",
        "storage_dirs": {
            "data": str(settings.data_dir),
            "models": str(settings.models_dir),
            "experiments": str(settings.experiments_dir),
        },
    }

