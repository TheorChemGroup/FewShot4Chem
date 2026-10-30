"""Abstract base class for evaluation heads."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np


class Head(ABC):
    """Evaluation head ABC.

    Subclasses dispatch on self.task (regression | classification) read from
    head_config.objective_type at __init__.
    """

    device: str = "cpu"
    task: Literal["regression", "classification"]

    @abstractmethod
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        **kwargs,
    ) -> None: ...

    @abstractmethod
    def predict(self, x_test: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray | None:  # noqa: ARG002
        """Return class probabilities for classifiers; None for regressors."""
        return None

    @abstractmethod
    def save(self, path: Path) -> None: ...
