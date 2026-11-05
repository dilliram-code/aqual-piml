"""
HydroPhysics AI: Backend Configuration and Environment Settings.
"""

import os
from pathlib import Path
from typing import List


class Settings:
    def __init__(self):
        self.app_name: str = os.getenv("APP_NAME", "HydroPhysics AI")
        self.app_env: str = os.getenv("APP_ENV", "development")
        self.port: int = int(os.getenv("API_PORT", "8000"))
        self.host: str = os.getenv("API_HOST", "0.0.0.0")

        # Data & Model storage paths
        self.base_dir: Path = Path(__file__).resolve().parent.parent
        self.data_dir: Path = self.base_dir / os.getenv("DATA_DIR", "data")
        self.camels_gb_dir: Path = self.base_dir / os.getenv("CAMELS_GB_DIR", "data/sample_camels_gb")
        self.models_dir: Path = self.base_dir / os.getenv("MODELS_DIR", "models_storage")
        self.experiments_dir: Path = self.base_dir / os.getenv("EXPERIMENTS_DIR", "experiments/results")

        # Ensure essential directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # CORS
        raw_origins = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000,*",
        )
        self.cors_origins: List[str] = [o.strip() for o in raw_origins.split(",") if o.strip()]

        # Machine Learning Defaults
        self.default_lookback: int = int(os.getenv("DEFAULT_LOOKBACK", "30"))
        self.default_horizon: int = int(os.getenv("DEFAULT_HORIZON", "1"))
        self.default_hidden_dim: int = int(os.getenv("DEFAULT_HIDDEN_DIM", "64"))
        self.default_epochs: int = int(os.getenv("DEFAULT_EPOCHS", "15"))
        self.default_batch_size: int = int(os.getenv("DEFAULT_BATCH_SIZE", "32"))
        self.default_lambda_physics: float = float(os.getenv("DEFAULT_PHYSICS_WEIGHT", "0.1"))
        self.random_seed: int = int(os.getenv("RANDOM_SEED", "42"))


settings = Settings()

