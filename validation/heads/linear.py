"""LinearHead — Ridge (regression) / LogisticRegression (classification) with internal CV."""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegressionCV, RidgeCV

from validation.heads._register_head import register_head_type
from validation.heads.abstract import Head

logger = logging.getLogger(__name__)


@register_head_type("linear")
class LinearHead(Head):
    """RidgeCV for regression, LogisticRegressionCV for classification."""

    def __init__(self, head_config: OmegaConf) -> None:
        self.head_config = head_config
        self.task = head_config.objective_type
        self.device = "cpu"
        self._model = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        x_val: np.ndarray | None = None,  # noqa: ARG002
        y_val: np.ndarray | None = None,  # noqa: ARG002
        sample_weight: np.ndarray | None = None,
        **_: Any,
    ) -> None:
        if self.task == "regression":
            self._model = RidgeCV(
                alphas=tuple(self.head_config.get("alphas", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])),
                cv=int(self.head_config.get("cv", 5)),
            )
            self._model.fit(x_train, y_train, sample_weight=sample_weight)
        else:
            self._model = LogisticRegressionCV(
                Cs=list(self.head_config.get("Cs", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])),
                cv=int(self.head_config.get("cv", 5)),
                max_iter=int(self.head_config.get("max_iter", 2000)),
                penalty="l2",
            )
            self._model.fit(x_train, y_train.ravel(), sample_weight=sample_weight)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LinearHead.predict called before fit")
        return self._model.predict(x_test)

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray | None:
        if self.task != "classification" or self._model is None:
            return None
        return self._model.predict_proba(x_test)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("LinearHead.save called before fit")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self._model, f)
