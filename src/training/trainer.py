"""
HydroPhysics AI: Training Pipeline Orchestrator and Checkpoint Storage.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional
import pandas as pd
from torch.utils.data import DataLoader

from src.data.camels_loader import CamelsGBLoader
from src.preprocessing.sequence_builder import (
    apply_data_scarcity,
    build_feature_matrices_and_datasets,
    chronological_split,
    inject_measurement_noise,
)


def run_experiment_task(
    config: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Worker task executed in background thread pool.
    Loads data, trains models according to config, and computes benchmark results.
    """
    gauge_id = str(config.get("gauge_id", "54057"))
    data_dir = config.get("data_dir", "data/sample_camels_gb")
    proportion = float(config.get("data_proportion", 1.0))
    noise_level = float(config.get("noise_level", 0.0))
    lambda_physics = float(config.get("lambda_physics", 0.1))
    lookback = int(config.get("lookback_window", 30))
    epochs = int(config.get("epochs", 15))
    batch_size = int(config.get("batch_size", 32))
    hidden_dim = int(config.get("hidden_dim", 64))

    loader = CamelsGBLoader(data_dir=data_dir)
    df = loader.load_catchment_timeseries(gauge_id)
    meta = loader.get_catchment_metadata(gauge_id)
    area = float(meta.get("area", 100.0))

    runner = BenchmarkRunner(
        lookback_window=lookback,
        forecast_horizon=1,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        lambda_physics=lambda_physics,
    )

    # Progress tracking adapter
    def runner_progress(msg, pct):
        if progress_callback:
            ep = int((pct / 100) * epochs)
            progress_callback(ep, epochs, None, None)

    results = runner.run_catchment_benchmark(
        df=df,
        catchment_area_km2=area,
        data_proportion=proportion,
        noise_level=noise_level,
        progress_callback=runner_progress,
    )

    results["metadata"]["gauge_id"] = gauge_id
    results["metadata"]["gauge_name"] = meta.get("gauge_name", f"Catchment {gauge_id}")
    results["metadata"]["regime"] = meta.get("regime", "Mixed")

    return results
