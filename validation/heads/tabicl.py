"""TabICLHead — in-context tabular model with per-target wrapping for multi-output.

TabICL classes verified on HPC: TabICLClassifier, TabICLRegressor (tabicl 2.1.1).
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.multioutput import MultiOutputRegressor

from validation.heads._register_head import register_head_type
from validation.heads.abstract import Head

logger = logging.getLogger(__name__)


@register_head_type("tabicl")
class TabICLHead(Head):
    """Per-target wrapping over TabICL for the multi-target case."""

    def __init__(self, head_config: OmegaConf) -> None:
        self.head_config = head_config
        self.task = head_config.objective_type
        self.device = head_config.get("device", "cpu") if torch.cuda.is_available() else "cpu"
        self._model = None
        self._is_multi = False

    def _make_estimator(self):  # noqa: ANN202
        if self.task == "regression":
            from tabicl import TabICLRegressor
            return TabICLRegressor(device=self.device, random_state=int(self.head_config.seed))
        from tabicl import TabICLClassifier
        return TabICLClassifier(device=self.device, random_state=int(self.head_config.seed))

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
        if sample_weight is not None:
            logger.warning(
                "TabICL does not support sample_weight; ignoring. "
                "Classification metrics on imbalanced datasets may be biased.",
            )
        y_arr = np.asarray(y_train)
        if y_arr.ndim == 2 and y_arr.shape[1] > 1 and self.task == "regression":
            self._model = MultiOutputRegressor(self._make_estimator())
            self._is_multi = True
            self._model.fit(x_train, y_arr)
            return
        self._model = self._make_estimator()
        self._is_multi = False
        self._model.fit(x_train, y_arr.ravel())

    # Mirror the TabPFN chunked-predict fix — large test splits (20k QM9 rows)
    # are likely to OOM on a single GPU call.
    _PREDICT_CHUNK_SIZE = 1024

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        if self._model is None:
            msg = "TabICLHead.predict called before fit"
            raise RuntimeError(msg)
        return self._chunked_predict(self._model.predict, x_test)

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray | None:
        if self.task != "classification" or self._model is None:
            return None
        return self._chunked_predict(self._model.predict_proba, x_test)

    def _chunked_predict(self, predict_fn, x_test: np.ndarray) -> np.ndarray:
        n = len(x_test)
        if n <= self._PREDICT_CHUNK_SIZE:
            return predict_fn(x_test)
        chunks = []
        for i in range(0, n, self._PREDICT_CHUNK_SIZE):
            chunks.append(predict_fn(x_test[i:i + self._PREDICT_CHUNK_SIZE]))
        return np.concatenate(chunks, axis=0)

    def save(self, path: Path) -> None:
        if self._model is None:
            msg = "TabICLHead.save called before fit"
            raise RuntimeError(msg)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self._model, f)
