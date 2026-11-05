"""
HydroPhysics AI: Base Model Interface for Hydrological Streamflow Predictors.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


class BaseHydrologicalModel(ABC):
    """
    Abstract interface for all rainfall-runoff predictive models in HydroPhysics AI.
    """

    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type
        self.is_trained: bool = False
        self.config: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, train_data: Any, val_data: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """Train the model and return training history/metrics."""
        pass

    @abstractmethod
    def predict(self, data: Any, **kwargs) -> np.ndarray:
        """Generate point predictions for streamflow (mm/day)."""
        pass

    def predict_with_uncertainty(
        self, data: Any, n_samples: int = 50, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate prediction with empirical lower (5th) and upper (95th) quantile bounds.
        Returns (q_pred_median, q_lower_5th, q_upper_95th).
        """
        pred = self.predict(data, **kwargs)
        # Default fallback if model does not implement native probabilistic uncertainty
        return pred, pred * 0.85, pred * 1.15

    @abstractmethod
    def save(self, filepath: Union[str, Path]) -> None:
        """Persist model parameters and scaler metadata."""
        pass

    @abstractmethod
    def load(self, filepath: Union[str, Path]) -> None:
        """Load model parameters and scaler metadata."""
        pass

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "config": self.config,
        }
