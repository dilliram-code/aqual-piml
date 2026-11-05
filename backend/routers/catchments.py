"""
HydroPhysics AI: Catchment Metadata, Attributes, and Discovery Router.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
import numpy as np

from backend.config import settings
from backend.schemas.api_schemas import CatchmentDetailResponse, CatchmentSummary
from src.data.camels_loader import CamelsGBLoader

router = APIRouter(prefix="/api/catchments", tags=["Catchments"])


@router.get("", response_model=List[CatchmentSummary])
def list_catchments():
    """List all available CAMELS-GB catchments with hydromorphological attributes."""
    loader = CamelsGBLoader(data_dir=settings.camels_gb_dir)
    attrs = loader.get_catchment_attributes()
    available_ids = loader.list_available_catchments()

    summaries = []
    for cid in available_ids:
        meta = loader.get_catchment_metadata(cid)
        summaries.append(
            CatchmentSummary(
                gauge_id=str(cid),
                gauge_name=str(meta.get("gauge_name", f"Catchment {cid}")),
                river=str(meta.get("river", "Unknown River")),
                area_km2=float(meta.get("area", 100.0)),
                regime=str(meta.get("regime", "Temperate Maritime")),
                gauge_lat=float(meta.get("gauge_lat", 52.0)),
                gauge_lon=float(meta.get("gauge_lon", -1.5)),
                elev_mean=float(meta.get("elev_mean", 150.0)),
                bfi_host=float(meta.get("bfi_host", 0.5)),
            )
        )
    return summaries


@router.get("/{gauge_id}", response_model=CatchmentDetailResponse)
def get_catchment_detail(gauge_id: str):
    """Retrieve detailed catchment attributes, observation coverage, and quality stats."""
    loader = CamelsGBLoader(data_dir=settings.camels_gb_dir)
    available_ids = loader.list_available_catchments()

    if str(gauge_id) not in available_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Catchment '{gauge_id}' not found. Available: {available_ids}",
        )

    meta = loader.get_catchment_metadata(gauge_id)
    df = loader.load_catchment_timeseries(gauge_id)

    # Calculate statistics
    date_range = {
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "total_days": len(df),
    }

    # Quality flags
    quality_counts = {}
    if "discharge_quality" in df.columns:
        quality_counts = df["discharge_quality"].value_counts().to_dict()

    data_summary = {
        "mean_precipitation_mm_day": round(float(df["precipitation"].mean()), 2),
        "mean_discharge_mm_day": round(float(df["discharge_spec"].mean()), 2),
        "mean_pet_mm_day": round(float(df["pet"].mean()), 2),
        "mean_temp_c": round(float(df["temperature"].mean()), 2),
        "runoff_ratio": round(float(df["discharge_spec"].sum() / max(1e-3, df["precipitation"].sum())), 3),
        "max_daily_discharge_mm": round(float(df["discharge_spec"].max()), 2),
        "max_daily_precipitation_mm": round(float(df["precipitation"].max()), 2),
    }

    return CatchmentDetailResponse(
        metadata=meta,
        data_summary=data_summary,
        date_range=date_range,
        quality_flag_distribution=quality_counts,
    )
