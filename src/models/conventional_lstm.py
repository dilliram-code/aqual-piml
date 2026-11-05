"""
HydroPhysics AI: Conventional LSTM Neural Network for Rainfall-Runoff Modeling.
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


class LSTMNetwork(nn.Module):
    """
    Standard PyTorch LSTM architecture with non-negative Softplus streamflow head.
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
        self.dropout_rate = dropout
        self.forecast_horizon = forecast_horizon

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: [batch_size, seq_len, input_dim]
        Returns:
            q_out: [batch_size, forecast_horizon] non-negative discharge
        """
        out, _ = self.lstm(x)  # out: [batch_size, seq_len, hidden_dim]
        last_hidden = out[:, -1, :]  # [batch_size, hidden_dim]
        dropped = self.dropout(last_hidden)
        raw_pred = self.fc(dropped)
        # Softplus ensures non-negative discharge strictly: Q >= 0
        q_out = F.softplus(raw_pred)
        return q_out


class ConventionalLSTMPredictor(BaseHydrologicalModel):
    """
    Wrapper for Conventional PyTorch LSTM with training, inference, and MC-Dropout uncertainty.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 1,
        learning_rate: float = 0.005,
        device: Optional[str] = None,
    ):
        super().__init__(name="Conventional LSTM", model_type="conventional_lstm")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.forecast_horizon = forecast_horizon
        self.learning_rate = learning_rate

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.network = LSTMNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizon=forecast_horizon,
        ).to(self.device)

        self.scaler: Optional[HydrologicalScaler] = None
        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "forecast_horizon": forecast_horizon,
            "learning_rate": learning_rate,
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
        criterion = nn.MSELoss()

        history: Dict[str, List[float]] = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
        }

        self.network.train()
        for epoch in range(1, epochs + 1):
            train_batch_losses = []
            for batch in train_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)

                optimizer.zero_grad()
                pred = self.network(x).view(-1)
                loss = criterion(pred, y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()
                train_batch_losses.append(loss.item())

            epoch_train_loss = float(np.mean(train_batch_losses))
            history["epochs"].append(epoch)
            history["train_loss"].append(epoch_train_loss)

            epoch_val_loss = None
            if val_loader is not None:
                self.network.eval()
                val_batch_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch["x"].to(self.device)
                        y = batch["y"].to(self.device)
                        pred = self.network(x).view(-1)
                        v_loss = criterion(pred, y.view(-1))
                        val_batch_losses.append(v_loss.item())
                epoch_val_loss = float(np.mean(val_batch_losses))
                history["val_loss"].append(epoch_val_loss)
                self.network.train()

            if progress_callback:
                progress_callback(epoch, epochs, epoch_train_loss, epoch_val_loss)

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
                out = self.network(x).squeeze(-1).cpu().numpy()
                preds.append(out)

        preds_arr = np.concatenate(preds, axis=0) if len(preds) > 0 else np.array([])
        if self.scaler is not None:
            # Rescale back to physical mm/day
            preds_arr = self.scaler.inverse_transform_target(preds_arr)

        return np.maximum(0.0, preds_arr)

    def predict_with_uncertainty(
        self,
        dataset_or_loader: Union[CatchmentDataset, DataLoader],
        n_samples: int = 50,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout (MCDO) epistemic uncertainty estimation.
        Keeps dropout active during inference to sample posterior predictive distribution.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict.")

        self.network.train()  # Keep dropout active!
        if isinstance(dataset_or_loader, CatchmentDataset):
            loader = DataLoader(dataset_or_loader, batch_size=64, shuffle=False)
        else:
            loader = dataset_or_loader

        all_sample_preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                sample_preds = []
                for batch in loader:
                    x = batch["x"].to(self.device)
                    out = self.network(x).squeeze(-1).cpu().numpy()
                    sample_preds.append(out)
                sample_arr = np.concatenate(sample_preds, axis=0)
                if self.scaler is not None:
                    sample_arr = self.scaler.inverse_transform_target(sample_arr)
                all_sample_preds.append(sample_arr)

        all_samples = np.array(all_sample_preds)  # [n_samples, n_timesteps]
        all_samples = np.maximum(0.0, all_samples)

        p50 = np.percentile(all_samples, 50, axis=0)
        p5 = np.percentile(all_samples, 5, axis=0)
        p95 = np.percentile(all_samples, 95, axis=0)

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
        if checkpoint.get("scaler"):
            self.scaler = HydrologicalScaler.from_dict(checkpoint["scaler"])
        self.is_trained = checkpoint.get("is_trained", True)
