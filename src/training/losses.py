"""
HydroPhysics AI: Physics-Informed Hydrological Loss Functions.

Mathematical Formulation:
--------------------------
The total loss is a composite of empirical prediction error and physical water-balance penalty:

    L_total = L_prediction + lambda_physics * L_water_balance

Where:
    - L_prediction: Mean Squared Error (or NSE-surrogate loss) between observed discharge
                    and model-predicted discharge in normalized target space.
    - L_water_balance: Mean squared mass conservation residual in physical units (mm/day)^2:
                    R_t = P_t - ET_t - Q_t - Delta S_t
                    L_water_balance = (1/N) * sum_i (R_i)^2
    - lambda_physics: Non-negative physical regularization weight parameter.

Physical Non-Negativity Constraints:
    - Q >= 0 enforced via Softplus activation
    - 0 <= ET <= PET enforced via Sigmoid fraction * PET
    - S >= 0 enforced via Softplus activation
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedLoss(nn.Module):
    """
    Composite loss function enforcing catchment mass conservation and streamflow accuracy.
    """

    def __init__(
        self,
        lambda_physics: float = 0.1,
        loss_type: str = "mse",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_physics = float(lambda_physics)
        self.loss_type = loss_type
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(
        self,
        q_pred_scaled: torch.Tensor,
        q_true_scaled: torch.Tensor,
        p_unscaled: torch.Tensor,
        pet_unscaled: torch.Tensor,
        et_pred_unscaled: torch.Tensor,
        q_pred_unscaled: torch.Tensor,
        delta_s_unscaled: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite loss and diagnostics.

        Parameters
        ----------
        q_pred_scaled : torch.Tensor
            Predicted discharge in normalized scale [B]
        q_true_scaled : torch.Tensor
            Observed discharge in normalized scale [B]
        p_unscaled : torch.Tensor
            Observed precipitation in mm/day [B]
        pet_unscaled : torch.Tensor
            Observed potential evapotranspiration in mm/day [B]
        et_pred_unscaled : torch.Tensor
            Model-estimated actual evapotranspiration in mm/day [B]
        q_pred_unscaled : torch.Tensor
            Model-predicted discharge in physical mm/day [B]
        delta_s_unscaled : torch.Tensor
            Model-estimated storage change in mm/day [B]

        Returns
        -------
        total_loss : torch.Tensor
            Scalar loss for backpropagation
        diagnostics : Dict[str, float]
            Individual loss components and physical metrics
        """
        # Ensure matching 1D tensor shapes
        q_pred_s = q_pred_scaled.view(-1)
        q_true_s = q_true_scaled.view(-1)
        p = p_unscaled.view(-1)
        pet = pet_unscaled.view(-1)
        et = et_pred_unscaled.view(-1)
        q_phys = q_pred_unscaled.view(-1)
        ds = delta_s_unscaled.view(-1)

        # 1. Prediction Loss (Normalized Space)
        l_pred = self.mse(q_pred_s, q_true_s)

        # 2. Physics Mass Conservation Loss (Physical Space: mm/day)
        # Residual R = P - ET - Q - Delta S
        residual = p - et - q_phys - ds
        l_wb = torch.mean(residual ** 2)

        # Optional soft barrier penalty if ET ever slightly exceeded PET (safety guard)
        et_excess = F.relu(et - pet)
        l_et_bound = torch.mean(et_excess ** 2)

        # 3. Composite Total Loss
        total_loss = l_pred + self.lambda_physics * l_wb + 1.0 * l_et_bound

        # Diagnostics for logging
        diagnostics = {
            "loss_total": float(total_loss.detach().item()),
            "loss_prediction": float(l_pred.detach().item()),
            "loss_water_balance": float(l_wb.detach().item()),
            "mean_abs_residual_mm": float(torch.mean(torch.abs(residual)).detach().item()),
            "mean_et_pred_mm": float(torch.mean(et).detach().item()),
            "mean_q_pred_mm": float(torch.mean(q_phys).detach().item()),
            "lambda_physics": self.lambda_physics,
        }

        return total_loss, diagnostics


def hand_calculated_residual(
    p: float,
    et: float,
    q: float,
    delta_s: float,
) -> float:
    """
    Pure reference function for exact analytical validation in unit tests.
    R = P - ET - Q - Delta S
    """
    return float(p - et - q - delta_s)
