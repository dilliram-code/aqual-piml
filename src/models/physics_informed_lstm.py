"""
HydroPhysics AI: Physics-Informed LSTM (PI-LSTM) with Coupled Catchment Water-Balance.

Architecture & Physical Invariants:
-----------------------------------
1. Input Layer: Scaled meteorological forcing vectors [P_scaled, PET_scaled, T_scaled].
2. Recurrent Core: Multi-layer LSTM extracting temporal storage and runoff dynamics.
3. Physical Flux Heads:
   - Streamflow Head: Q_pred = Softplus(Linear(h)) >= 0
   - Actual Evapotranspiration Head: alpha = Sigmoid(Linear(h)) in (0, 1)
                                      ET_pred = alpha * PET_physical  [Guarantees 0 <= ET <= PET]
   - Moisture Storage Head: S_t = Softplus(Linear(h)) >= 0
                            Delta_S_t = S_t - S_{t-1}
4. Regularization:
   L_total = L_MSE + lambda_physics * L_WaterBalance
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.base import BaseHydrologicalModel
from src.preprocessing.scaler import HydrologicalScaler
from src.preprocessing.sequence_builder import CatchmentDataset
from src.training.losses import PhysicsInformedLoss


class PhysicsInformedLSTMNetwork(nn.Module):
    """
    PyTorch Neural Network coupling LSTM representations with physical mass conservation heads.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Head 1: Streamflow discharge
        self.fc_q = nn.Linear(hidden_dim, forecast_horizon)

        # Head 2: Actual Evapotranspiration fraction (alpha in [0, 1])
        self.fc_et_frac = nn.Linear(hidden_dim, forecast_horizon)

        # Head 3: Latent Storage state S_t
        self.fc_storage = nn.Linear(hidden_dim, forecast_horizon)

        # Head 4: Storage change rate Delta S
        self.fc_delta_s = nn.Linear(hidden_dim, forecast_horizon)

    def forward(
        self,
        x: torch.Tensor,
        pet_physical: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass yielding physical fluxes and states.
        """
        out, _ = self.lstm(x)  # [B, seq_len, hidden_dim]
        last_h = out[:, -1, :]  # [B, hidden_dim]
        dropped = self.dropout(last_h)

        # 1. Non-negative streamflow in normalized scale
        q_scaled = F.softplus(self.fc_q(dropped)).squeeze(-1)

        # 2. Actual ET fraction: bounded strictly in [0, 1] by Sigmoid
        et_fraction = torch.sigmoid(self.fc_et_frac(dropped)).squeeze(-1)

        # If physical PET is provided, compute physical ET flux
        et_physical = None
        if pet_physical is not None:
            et_physical = et_fraction * pet_physical

        # 3. Storage state S >= 0
        storage_state = F.softplus(self.fc_storage(dropped)).squeeze(-1)

        # 4. Storage change Delta S (can be positive or negative)
        delta_s = self.fc_delta_s(dropped).squeeze(-1)

        return {
            "q_scaled": q_scaled,
            "et_fraction": et_fraction,
            "et_physical": et_physical,
            "storage_state": storage_state,
            "delta_s": delta_s,
        }


class PhysicsInformedLSTMPredictor(BaseHydrologicalModel):
    """
    High-level Physics-Informed LSTM regressor integrating physical loss,
    training routines, MC-Dropout uncertainty, and state persistence.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 1,
        learning_rate: float = 0.005,
        lambda_physics: float = 0.1,
        device: Optional[str] = None,
    ):
        super().__init__(name="Physics-Informed LSTM", model_type="physics_informed_lstm")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.forecast_horizon = forecast_horizon
        self.learning_rate = learning_rate
        self.lambda_physics = lambda_physics

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.network = PhysicsInformedLSTMNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizon=forecast_horizon,
        ).to(self.device)

        self.scaler: Optional[HydrologicalScaler] = None
        self.loss_fn = PhysicsInformedLoss(lambda_physics=lambda_physics)
        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "forecast_horizon": forecast_horizon,
            "learning_rate": learning_rate,
            "lambda_physics": lambda_physics,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 20,
        scaler: Optional[HydrologicalScaler] = None,
        progress_callback: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        self.scaler = scaler
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

        history: Dict[str, List[float]] = {
            "epochs": [],
            "train_loss": [],
            "train_pred_loss": [],
            "train_physics_loss": [],
            "val_loss": [],
            "mean_abs_residual_mm": [],
        }

        self.network.train()
        for epoch in range(1, epochs + 1):
            batch_totals = []
            batch_preds = []
            batch_phys = []
            batch_res = []

            for batch in train_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                p_unscaled = batch["p_target_unscaled"].to(self.device)
                pet_unscaled = batch["pet_target_unscaled"].to(self.device)

                optimizer.zero_grad()
                out = self.network(x, pet_physical=pet_unscaled)
                q_scaled = out["q_scaled"]
                et_physical = out["et_physical"]
                delta_s = out["delta_s"]

                # Convert predicted q_scaled to physical mm/day for mass conservation loss
                if self.scaler is not None and "discharge_spec" in self.scaler.stats:
                    stat = self.scaler.stats["discharge_spec"]
                    if self.scaler.method == "standard":
                        q_unscaled = q_scaled * stat["std"] + stat["mean"]
                    else:
                        q_unscaled = q_scaled * (stat["max"] - stat["min"]) + stat["min"]
                else:
                    q_unscaled = q_scaled

                total_loss, diag = self.loss_fn(
                    q_pred_scaled=q_scaled,
                    q_true_scaled=y,
                    p_unscaled=p_unscaled,
                    pet_unscaled=pet_unscaled,
                    et_pred_unscaled=et_physical,
                    q_pred_unscaled=q_unscaled,
                    delta_s_unscaled=delta_s,
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()

                batch_totals.append(diag["loss_total"])
                batch_preds.append(diag["loss_prediction"])
                batch_phys.append(diag["loss_water_balance"])
                batch_res.append(diag["mean_abs_residual_mm"])

            ep_total = float(np.mean(batch_totals))
            ep_pred = float(np.mean(batch_preds))
            ep_phys = float(np.mean(batch_phys))
            ep_res = float(np.mean(batch_res))

            history["epochs"].append(epoch)
            history["train_loss"].append(ep_total)
            history["train_pred_loss"].append(ep_pred)
            history["train_physics_loss"].append(ep_phys)
            history["mean_abs_residual_mm"].append(ep_res)

            ep_val = None
            if val_loader is not None:
                self.network.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch["x"].to(self.device)
                        y = batch["y"].to(self.device)
                        p_unscaled = batch["p_target_unscaled"].to(self.device)
                        pet_unscaled = batch["pet_target_unscaled"].to(self.device)

                        out = self.network(x, pet_physical=pet_unscaled)
                        q_scaled = out["q_scaled"]
                        et_physical = out["et_physical"]
                        delta_s = out["delta_s"]

                        if self.scaler is not None and "discharge_spec" in self.scaler.stats:
                            stat = self.scaler.stats["discharge_spec"]
                            if self.scaler.method == "standard":
                                q_unscaled = q_scaled * stat["std"] + stat["mean"]
                            else:
                                q_unscaled = q_scaled * (stat["max"] - stat["min"]) + stat["min"]
                        else:
                            q_unscaled = q_scaled

                        v_loss, _ = self.loss_fn(
                            q_pred_scaled=q_scaled,
                            q_true_scaled=y,
                            p_unscaled=p_unscaled,
                            pet_unscaled=pet_unscaled,
                            et_pred_unscaled=et_physical,
                            q_pred_unscaled=q_unscaled,
                            delta_s_unscaled=delta_s,
                        )
                        val_losses.append(v_loss.item())

                ep_val = float(np.mean(val_losses))
                history["val_loss"].append(ep_val)
                self.network.train()

            if progress_callback:
                progress_callback(epoch, epochs, ep_total, ep_val)

        self.is_trained = True
        return history

    def predict(self, dataset_or_loader: Union[CatchmentDataset, DataLoader], **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict.")

        self.network.eval()
        if isinstance(dataset_or_loader, CatchmentDataset):
            loader = DataLoader(dataset_or_loader, batch_size=64, shuffle=False)
        else:
            loader = dataset_or_loader

        preds = []
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                pet_unscaled = batch["pet_target_unscaled"].to(self.device)
                out = self.network(x, pet_physical=pet_unscaled)
                preds.append(out["q_scaled"].cpu().numpy())

        preds_arr = np.concatenate(preds, axis=0) if len(preds) > 0 else np.array([])
        if self.scaler is not None:
            preds_arr = self.scaler.inverse_transform_target(preds_arr)

        return np.maximum(0.0, preds_arr)

    def predict_detailed(
        self,
        dataset_or_loader: Union[CatchmentDataset, DataLoader],
    ) -> Dict[str, np.ndarray]:
        """
        Generate streamflow predictions alongside latent actual ET and storage change states.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict.")

        self.network.eval()
        if isinstance(dataset_or_loader, CatchmentDataset):
            loader = DataLoader(dataset_or_loader, batch_size=64, shuffle=False)
        else:
            loader = dataset_or_loader

        all_q = []
        all_et = []
        all_storage = []
        all_delta_s = []

        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                pet_unscaled = batch["pet_target_unscaled"].to(self.device)
                out = self.network(x, pet_physical=pet_unscaled)

                all_q.append(out["q_scaled"].cpu().numpy())
                all_et.append(out["et_physical"].cpu().numpy())
                all_storage.append(out["storage_state"].cpu().numpy())
                all_delta_s.append(out["delta_s"].cpu().numpy())

        q_arr = np.concatenate(all_q, axis=0)
        if self.scaler is not None:
            q_arr = self.scaler.inverse_transform_target(q_arr)

        et_arr = np.maximum(0.0, np.concatenate(all_et, axis=0))
        s_arr = np.maximum(0.0, np.concatenate(all_storage, axis=0))
        ds_arr = np.concatenate(all_delta_s, axis=0)

        return {
            "q_pred_mm": np.maximum(0.0, q_arr),
            "et_pred_mm": et_arr,
            "storage_mm": s_arr,
            "delta_s_mm": ds_arr,
        }

    def predict_with_uncertainty(
        self,
        dataset_or_loader: Union[CatchmentDataset, DataLoader],
        n_samples: int = 50,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout uncertainty sampling.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict.")

        self.network.train()  # Keep dropout active
        if isinstance(dataset_or_loader, CatchmentDataset):
            loader = DataLoader(dataset_or_loader, batch_size=64, shuffle=False)
        else:
            loader = dataset_or_loader

        all_samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                sample_preds = []
                for batch in loader:
                    x = batch["x"].to(self.device)
                    pet_unscaled = batch["pet_target_unscaled"].to(self.device)
                    out = self.network(x, pet_physical=pet_unscaled)
                    sample_preds.append(out["q_scaled"].cpu().numpy())
                sample_arr = np.concatenate(sample_preds, axis=0)
                if self.scaler is not None:
                    sample_arr = self.scaler.inverse_transform_target(sample_arr)
                all_samples.append(sample_arr)

        samples_mat = np.array(all_samples)
        samples_mat = np.maximum(0.0, samples_mat)

        p50 = np.percentile(samples_mat, 50, axis=0)
        p5 = np.percentile(samples_mat, 5, axis=0)
        p95 = np.percentile(samples_mat, 95, axis=0)

        self.network.eval()
        return p50, p5, p95

    def save(self, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.network.state_dict(),
            "config": self.config,
            "scaler": self.scaler.to_dict() if self.scaler else None,
            "is_trained": self.is_trained,
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: Union[str, Path]) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["state_dict"])
        self.config = checkpoint["config"]
        self.lambda_physics = self.config.get("lambda_physics", 0.1)
        self.loss_fn.lambda_physics = self.lambda_physics
        if checkpoint.get("scaler"):
            self.scaler = HydrologicalScaler.from_dict(checkpoint["scaler"])
        self.is_trained = checkpoint.get("is_trained", True)
