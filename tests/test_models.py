"""
Unit Tests: Machine Learning Models, Architectural Shapes, and Non-Negativity Guarantees.
"""

import pytest
import numpy as np
import torch

from src.models.conventional_lstm import LSTMNetwork
from src.models.physics_informed_lstm import PhysicsInformedLSTMNetwork
from src.models.random_forest import RandomForestHydrologyModel


def test_conventional_lstm_output_shape_and_non_negativity():
    """Conventional LSTM output must match forecast horizon and be strictly non-negative (Softplus)."""
    batch_size = 8
    seq_len = 30
    input_dim = 3
    horizon = 1

    net = LSTMNetwork(input_dim=input_dim, hidden_dim=32, num_layers=1, forecast_horizon=horizon)
    x = torch.randn(batch_size, seq_len, input_dim)

    out = net(x)
    assert out.shape == (batch_size, horizon)
    # Physical streamflow non-negativity: Q >= 0
    assert torch.all(out >= 0.0)


def test_pi_lstm_flux_bounds():
    """
    Physics-Informed LSTM must guarantee:
    1. Discharge Q >= 0 (Softplus)
    2. Actual ET fraction alpha in [0, 1] (Sigmoid)
    3. Storage state S >= 0 (Softplus)
    4. Actual ET <= PET strictly
    """
    batch_size = 16
    seq_len = 30
    net = PhysicsInformedLSTMNetwork(input_dim=3, hidden_dim=32, num_layers=1, forecast_horizon=1)

    x = torch.randn(batch_size, seq_len, 3)
    pet = torch.full((batch_size,), 4.5)  # 4.5 mm/day PET

    outputs = net(x, pet_physical=pet)

    assert torch.all(outputs["q_scaled"] >= 0.0)
    assert torch.all(outputs["et_fraction"] >= 0.0)
    assert torch.all(outputs["et_fraction"] <= 1.0)
    assert torch.all(outputs["storage_state"] >= 0.0)
    assert torch.all(outputs["et_physical"] <= pet + 1e-6)

