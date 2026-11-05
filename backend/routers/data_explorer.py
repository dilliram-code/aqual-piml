"""
HydroPhysics AI: Data Exploration, Hydrograph & Hyetograph Timeseries Router.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
import numpy as np
import pandas as pd

from backend.config import settings
from src.data.camels_loader import CamelsGBLoader

router = APIRouter(prefix="/api/data", tags=["Data Explorer"])


@router.get("/{gauge_id}")
def get_catchment_timeseries_data(
    gauge_id: str,
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    downsample_factor: int = Query(1, ge=1, le=30, description="Stride/downsample factor for fast rendering"),
):
    """Retrieve hydrometeorological timeseries data for interactive Plotly charts."""
    loader = CamelsGBLoader(data_dir=settings.camels_gb_dir)
    available = loader.list_available_catchments()
    if str(gauge_id) not in available:
        raise HTTPException(status_code=404, detail=f"Catchment {gauge_id} not found.")

    df = loader.load_catchment_timeseries(
        gauge_id=gauge_id,
        start_date=start_date,
        end_date=end_date,
    )

    if downsample_factor > 1:
        df_view = df.iloc[::downsample_factor]
    else:
        df_view = df

    dates = [str(d.date()) for d in df_view.index]

    return {
        "gauge_id": gauge_id,
        "count": len(df_view),
        "dates": dates,
        "precipitation": [round(float(v), 2) for v in df_view["precipitation"].values],
        "discharge_spec": [round(float(v), 3) for v in df_view["discharge_spec"].values],
        "discharge_vol": [round(float(v), 3) for v in df_view["discharge_vol"].values],
        "pet": [round(float(v), 2) for v in df_view["pet"].values],
        "temperature": [round(float(v), 2) for v in df_view["temperature"].values],
        "quality_flags": list(df_view["discharge_quality"].values) if "discharge_quality" in df_view.columns else [],
    }


@router.get("/{gauge_id}/stats")
def get_catchment_statistics(gauge_id: str):
    """Descriptive statistics, distribution percentiles, and missing data audit."""
    loader = CamelsGBLoader(data_dir=settings.camels_gb_dir)
    if str(gauge_id) not in loader.list_available_catchments():
        raise HTTPException(status_code=404, detail=f"Catchment {gauge_id} not found.")

    df = loader.load_catchment_timeseries(gauge_id, fill_missing=False)

    numeric_cols = ["precipitation", "pet", "temperature", "discharge_spec", "discharge_vol"]
    stats_out = {}

    for col in numeric_cols:
        if col in df.columns:
            series = df[col].dropna()
            stats_out[col] = {
                "count": int(len(series)),
                "missing_count": int(df[col].isna().sum()),
                "missing_pct": round(float(df[col].isna().mean() * 100), 2),
                "mean": round(float(series.mean()), 3),
                "std": round(float(series.std()), 3),
                "min": round(float(series.min()), 3),
                "p25": round(float(series.quantile(0.25)), 3),
                "median": round(float(series.median()), 3),
                "p75": round(float(series.quantile(0.75)), 3),
                "p95": round(float(series.quantile(0.95)), 3),
                "max": round(float(series.max()), 3),
            }

    return {
        "gauge_id": gauge_id,
        "statistics": stats_out,
        "total_records": len(df),
    }


@router.get("/{gauge_id}/correlations")
def get_cross_correlations(gauge_id: str):
    """Compute lag cross-correlations between precipitation and streamflow."""
    loader = CamelsGBLoader(data_dir=settings.camels_gb_dir)
    if str(gauge_id) not in loader.list_available_catchments():
        raise HTTPException(status_code=404, detail=f"Catchment {gauge_id} not found.")

    df = loader.load_catchment_timeseries(gauge_id)

    # Cross-correlation of P(t-k) with Q(t) for lags k = 0..14 days
    lags = list(range(15))
    corrs = []
    p = df["precipitation"]
    q = df["discharge_spec"]

    for lag in lags:
        p_lagged = p.shift(lag)
        valid = (~p_lagged.isna()) & (~q.isna())
        c = float(np.corrcoef(p_lagged[valid], q[valid])[0, 1])
        corrs.append(round(c, 4))

    return {
        "gauge_id": gauge_id,
        "lags_days": lags,
        "precipitation_discharge_correlations": corrs,
        "optimal_lag_days": int(np.argmax(corrs)),
    }
