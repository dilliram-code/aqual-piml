"""
Unit Tests: Hand-Calculated Physics Loss Verification and Gradient Flow.
"""

import pytest
import numpy as np
import torch

from src.training.losses import PhysicsInformedLoss, hand_calculated_residual


def test_hand_calculated_loss_components():
    """
    Exact analytical hand-calculation test:
    Given:
        q_pred_scaled = [1.0, 2.0]
        q_true_scaled = [1.5, 2.5]
        -> L_pred = ((1.0 - 1.5)^2 + (2.0 - 2.5)^2) / 2 = (0.25 + 0.25) / 2 = 0.25

        p = [10.0, 8.0]
        et = [3.0, 2.0]
        q_phys = [4.0, 3.0]
        delta_s = [1.0, 2.0]
        Residuals:
            R1 = 10.0 - 3.0 - 4.0 - 1.0 = 2.0 -> R1^2 = 4.0
            R2 = 8.0 - 2.0 - 3.0 - 2.0 = 1.0  -> R2^2 = 1.0
        -> L_wb = (4.0 + 1.0) / 2 = 2.5

        lambda_physics = 0.2
        Expected L_total = 0.25 + 0.2 * 2.5 = 0.25 + 0.50 = 0.75
    """
    loss_fn = PhysicsInformedLoss(lambda_physics=0.2)

    q_pred_s = torch.tensor([1.0, 2.0], requires_grad=True)
    q_true_s = torch.tensor([1.5, 2.5])
    p = torch.tensor([10.0, 8.0])
    pet = torch.tensor([5.0, 4.0])
    et = torch.tensor([3.0, 2.0], requires_grad=True)
    q_phys = torch.tensor([4.0, 3.0], requires_grad=True)
    ds = torch.tensor([1.0, 2.0], requires_grad=True)

    total_loss, diag = loss_fn(
        q_pred_scaled=q_pred_s,
        q_true_scaled=q_true_s,
        p_unscaled=p,
        pet_unscaled=pet,
        et_pred_unscaled=et,
        q_pred_unscaled=q_phys,
        delta_s_unscaled=ds,
    )

    assert np.isclose(diag["loss_prediction"], 0.25)
    assert np.isclose(diag["loss_water_balance"], 2.50)
    assert np.isclose(float(total_loss.item()), 0.75)

    # Test individual analytical helper
    assert np.isclose(hand_calculated_residual(10.0, 3.0, 4.0, 1.0), 2.0)
    assert np.isclose(hand_calculated_residual(8.0, 2.0, 3.0, 2.0), 1.0)


def test_loss_gradient_flow():
    """Verify that gradients properly backpropagate through physical mass balance terms."""
    loss_fn = PhysicsInformedLoss(lambda_physics=0.5)

    q_pred_s = torch.tensor([1.0], requires_grad=True)
    q_true_s = torch.tensor([1.0])
    p = torch.tensor([10.0])
    pet = torch.tensor([5.0])
    et = torch.tensor([2.0], requires_grad=True)
    q_phys = torch.tensor([2.0], requires_grad=True)
    ds = torch.tensor([1.0], requires_grad=True)

    # Note: q_pred_s matches q_true_s (pred loss is 0), so gradient comes solely from physics loss!
    total_loss, _ = loss_fn(q_pred_s, q_true_s, p, pet, et, q_phys, ds)
    total_loss.backward()

    # Residual = 10 - 2 - 2 - 1 = 5.0
    # L_wb = 5.0^2 = 25.0
    # d(L_total)/d(q_phys) = lambda * 2 * R * (-1) = 0.5 * 2 * 5 * (-1) = -5.0
    assert q_phys.grad is not None
    assert np.isclose(float(q_phys.grad.item()), -5.0)
    assert et.grad is not None
    assert np.isclose(float(et.grad.item()), -5.0)
