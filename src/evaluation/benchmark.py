"""
HydroPhysics AI: Multi-Model Benchmark Runner and Comparison Engine.

Strict Research Protocol:
-------------------------
1. Identical Chronological Splits: All compared models are evaluated on the EXACT same test window.
2. Identical Forcing Inputs: All models receive identical meteorological variables (P, PET, T).
3. Fair Metric Computation: Consistent unscaled physical units (mm/day and m^3/s).
4. Wall-Clock Latency Profiling: Measures training duration and inference latency per sample.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_comprehensive_metrics
from src.models.conventional_lstm import ConventionalLSTMPredictor
from src.models.physics_informed_lstm import PhysicsInformedLSTMPredictor
from src.models.random_forest import RandomForestHydrologyModel
from src.physics.unit_conversions import runoff_mm_day_to_m3s
from src.preprocessing.sequence_builder import (
    apply_data_scarcity,
    build_feature_matrices_and_datasets,
    chronological_split,
    inject_measurement_noise,
)


class BenchmarkRunner:
    """
    Orchestrates rigorous, reproducible comparative experiments across
    Random Forest, Conventional LSTM, and Physics-Informed LSTM.
    """

    def __init__(
        self,
        lookback_window: int = 30,
        forecast_horizon: int = 1,
        epochs: int = 15,
        batch_size: int = 32,
        hidden_dim: int = 64,
        lambda_physics: float = 0.1,
        random_seed: int = 42,
    ):
        self.lookback = lookback_window
        self.horizon = forecast_horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.lambda_physics = lambda_physics
        self.seed = random_seed

    def run_catchment_benchmark(
        self,
        df: pd.DataFrame,
        catchment_area_km2: float = 100.0,
        data_proportion: float = 1.0,
        noise_level: float = 0.0,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute full benchmark on a catchment timeseries.
        """
        # 1. Chronological Splitting
        train_df, val_df, test_df = chronological_split(df, train_ratio=0.65, val_ratio=0.15)

        # 2. Data Scarcity Subsetting (applied to training set only)
        if data_proportion < 1.0:
            train_df = apply_data_scarcity(train_df, proportion=data_proportion, seed=self.seed)

        # 3. Measurement Noise Injection (if configured for noise robustness testing)
        if noise_level > 0.0:
            train_df = inject_measurement_noise(train_df, noise_level=noise_level, seed=self.seed)

        # 4. Feature preprocessing and dataset building
        train_ds, val_ds, test_ds, scaler = build_feature_matrices_and_datasets(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            lookback_window=self.lookback,
            forecast_horizon=self.horizon,
        )

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        # Extract Ground Truth for test window
        y_test_true = test_df["discharge_spec"].values[self.lookback :]
        test_dates = test_df.index[self.lookback :]
        p_test = test_df["precipitation"].values[self.lookback :]
        pet_test = test_df["pet"].values[self.lookback :]

        results: Dict[str, Any] = {
            "metadata": {
                "catchment_area_km2": catchment_area_km2,
                "data_proportion": data_proportion,
                "noise_level": noise_level,
                "lambda_physics": self.lambda_physics,
                "test_period_days": len(y_test_true),
                "test_start_date": str(test_dates[0].date()) if len(test_dates) > 0 else None,
                "test_end_date": str(test_dates[-1].date()) if len(test_dates) > 0 else None,
            },
            "models": {},
            "predictions": {
                "dates": [str(d.date()) for d in test_dates],
                "observed_mm": [round(float(v), 3) for v in y_test_true],
                "precipitation_mm": [round(float(v), 2) for v in p_test],
                "pet_mm": [round(float(v), 2) for v in pet_test],
            },
        }

        # ---------------------------------------------------------------------
        # MODEL 1: Random Forest Baseline
        # ---------------------------------------------------------------------
        if progress_callback:
            progress_callback("Training Random Forest baseline...", 10)

        rf_model = RandomForestHydrologyModel(n_estimators=100, max_depth=12, random_state=self.seed)
        t0 = time.time()
        rf_model.fit(train_df, val_df)
        rf_train_time = round(time.time() - t0, 3)

        t0 = time.time()
        rf_preds = rf_model.predict(test_df)[self.lookback :]
        rf_p50, rf_p5, rf_p95 = rf_model.predict_with_uncertainty(test_df)
        rf_p5 = rf_p5[self.lookback :]
        rf_p95 = rf_p95[self.lookback :]
        rf_infer_time_ms = round(1000.0 * (time.time() - t0) / max(1, len(y_test_true)), 3)

        rf_metrics = compute_comprehensive_metrics(
            y_true=y_test_true,
            y_pred=rf_preds,
            q_lower=rf_p5,
            q_upper=rf_p95,
        )
        rf_metrics["train_time_sec"] = rf_train_time
        rf_metrics["infer_time_ms_per_sample"] = rf_infer_time_ms

        results["models"]["random_forest"] = {
            "name": "Random Forest Baseline",
            "metrics": rf_metrics,
        }
        results["predictions"]["rf_pred_mm"] = [round(float(v), 3) for v in rf_preds]
        results["predictions"]["rf_lower_mm"] = [round(float(v), 3) for v in rf_p5]
        results["predictions"]["rf_upper_mm"] = [round(float(v), 3) for v in rf_p95]

        # ---------------------------------------------------------------------
        # MODEL 2: Conventional LSTM
        # ---------------------------------------------------------------------
        if progress_callback:
            progress_callback("Training Conventional LSTM...", 40)

        lstm_model = ConventionalLSTMPredictor(
            input_dim=3,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            dropout=0.2,
            forecast_horizon=self.horizon,
            learning_rate=0.005,
        )
        t0 = time.time()
        lstm_model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.epochs,
            scaler=scaler,
        )
        lstm_train_time = round(time.time() - t0, 3)

        t0 = time.time()
        lstm_preds = lstm_model.predict(test_loader)
        lstm_p50, lstm_p5, lstm_p95 = lstm_model.predict_with_uncertainty(test_loader, n_samples=30)
        lstm_infer_time_ms = round(1000.0 * (time.time() - t0) / max(1, len(y_test_true)), 3)

        lstm_metrics = compute_comprehensive_metrics(
            y_true=y_test_true,
            y_pred=lstm_preds,
            q_lower=lstm_p5,
            q_upper=lstm_p95,
        )
        lstm_metrics["train_time_sec"] = lstm_train_time
        lstm_metrics["infer_time_ms_per_sample"] = lstm_infer_time_ms

        results["models"]["conventional_lstm"] = {
            "name": "Conventional LSTM",
            "metrics": lstm_metrics,
        }
        results["predictions"]["lstm_pred_mm"] = [round(float(v), 3) for v in lstm_preds]
        results["predictions"]["lstm_lower_mm"] = [round(float(v), 3) for v in lstm_p5]
        results["predictions"]["lstm_upper_mm"] = [round(float(v), 3) for v in lstm_p95]

        # ---------------------------------------------------------------------
        # MODEL 3: Physics-Informed LSTM (PI-LSTM)
        # ---------------------------------------------------------------------
        if progress_callback:
            progress_callback("Training Physics-Informed LSTM...", 75)

        pi_lstm = PhysicsInformedLSTMPredictor(
            input_dim=3,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            dropout=0.2,
            forecast_horizon=self.horizon,
            learning_rate=0.005,
            lambda_physics=self.lambda_physics,
        )
        t0 = time.time()
        pi_history = pi_lstm.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.epochs,
            scaler=scaler,
        )
        pi_train_time = round(time.time() - t0, 3)

        t0 = time.time()
        detailed_preds = pi_lstm.predict_detailed(test_loader)
        pi_preds = detailed_preds["q_pred_mm"]
        et_preds = detailed_preds["et_pred_mm"]
        storage_preds = detailed_preds["storage_mm"]
        delta_s_preds = detailed_preds["delta_s_mm"]

        pi_p50, pi_p5, pi_p95 = pi_lstm.predict_with_uncertainty(test_loader, n_samples=30)
        pi_infer_time_ms = round(1000.0 * (time.time() - t0) / max(1, len(y_test_true)), 3)

        pi_metrics = compute_comprehensive_metrics(
            y_true=y_test_true,
            y_pred=pi_preds,
            p_series=p_test,
            et_series=et_preds,
            delta_s_series=delta_s_preds,
            q_lower=pi_p5,
            q_upper=pi_p95,
        )
        pi_metrics["train_time_sec"] = pi_train_time
        pi_metrics["infer_time_ms_per_sample"] = pi_infer_time_ms

        results["models"]["physics_informed_lstm"] = {
            "name": "Physics-Informed LSTM",
            "metrics": pi_metrics,
            "training_history": pi_history,
        }
        results["predictions"]["pi_lstm_pred_mm"] = [round(float(v), 3) for v in pi_preds]
        results["predictions"]["pi_lstm_lower_mm"] = [round(float(v), 3) for v in pi_p5]
        results["predictions"]["pi_lstm_upper_mm"] = [round(float(v), 3) for v in pi_p95]
        results["predictions"]["et_pred_mm"] = [round(float(v), 3) for v in et_preds]
        results["predictions"]["storage_mm"] = [round(float(v), 3) for v in storage_preds]
        results["predictions"]["delta_s_mm"] = [round(float(v), 3) for v in delta_s_preds]

        # Calculate volumetric predictions (m^3/s) for display
        results["predictions"]["observed_m3s"] = [
            round(float(v), 3) for v in runoff_mm_day_to_m3s(y_test_true, catchment_area_km2)
        ]
        results["predictions"]["pi_lstm_pred_m3s"] = [
            round(float(v), 3) for v in runoff_mm_day_to_m3s(pi_preds, catchment_area_km2)
        ]

        if progress_callback:
            progress_callback("Benchmark complete!", 100)

        return results
