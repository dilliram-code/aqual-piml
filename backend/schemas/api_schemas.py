"""
HydroPhysics AI: Pydantic Data Contracts and API Schemas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    app_name: str = "HydroPhysics AI"
    dataset_status: str
    available_catchments: int
    models_ready: bool


class CatchmentSummary(BaseModel):
    gauge_id: str
    gauge_name: str
    river: Optional[str] = None
    area_km2: float
    regime: str
    gauge_lat: float
    gauge_lon: float
    elev_mean: Optional[float] = None
    bfi_host: Optional[float] = None


class CatchmentDetailResponse(BaseModel):
    metadata: Dict[str, Any]
    data_summary: Dict[str, Any]
    date_range: Dict[str, Any]
    quality_flag_distribution: Dict[str, int]


class LaunchExperimentRequest(BaseModel):
    gauge_id: str = Field(default="54057", description="CAMELS-GB Catchment Gauge ID")
    data_proportion: float = Field(default=1.0, ge=0.05, le=1.0, description="Fraction of training observations (e.g. 0.25 for 25%)")
    noise_level: float = Field(default=0.0, ge=0.0, le=0.50, description="Relative measurement noise standard deviation")
    lambda_physics: float = Field(default=0.1, ge=0.0, le=10.0, description="Physical water-balance loss weight lambda")
    lookback_window: int = Field(default=30, ge=5, le=120, description="Sequence lookback window in days")
    epochs: int = Field(default=15, ge=1, le=100, description="Training epochs for LSTM models")
    batch_size: int = Field(default=32, ge=8, le=256, description="Training mini-batch size")
    hidden_dim: int = Field(default=64, ge=16, le=256, description="Hidden dimensions for LSTM cells")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress_pct: int
    current_epoch: int
    total_epochs: int
    current_train_loss: Optional[float] = None
    current_val_loss: Optional[float] = None
    status_message: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    has_results: bool


class PredictRequest(BaseModel):
    gauge_id: str = "54057"
    model_type: str = Field(default="physics_informed_lstm", description="random_forest, conventional_lstm, or physics_informed_lstm")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    lookback_window: int = 30


class PhysicsDiagnosticsResponse(BaseModel):
    equation: str
    variables: Dict[str, str]
    units: Dict[str, str]
    assumptions: List[str]
    limitations: List[str]
    summary_metrics: Dict[str, Any]
