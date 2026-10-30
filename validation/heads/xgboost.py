"""XGBoostHead — port of the legacy XGBoost path into the Head abstraction."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBModel, XGBRegressor

from validation.heads._register_head import register_head_type
from validation.heads.abstract import Head

logger = logging.getLogger(__name__)


@register_head_type("xgboost")
class XGBoostHead(Head):
    """Wraps XGBoost with GPU-tensor branch and optional eval_set early stopping.

    When head_config.params.early_stopping_rounds is set, 10% of training data
    is carved out as a validation set for early stopping. Otherwise training uses
    the full x_train (matching legacy TargetValidator behavior).
    For classification, sample_weight is auto-computed from class frequencies
    when not explicitly provided (porting legacy calculate_sample_weights behavior).
    """

    def __init__(self, head_config: OmegaConf) -> None:
        self.head_config = head_config
        self.task = head_config.objective_type
        self.device = head_config.params.get("device", "cpu")
        self._random_seed = int(head_config.params.get("seed", 42))
        self._model: XGBModel | None = None

    def _make_xgb_model(self) -> XGBModel:
        if self.task == "regression":
            return XGBRegressor(**self.head_config.params)
        return XGBClassifier(**self.head_config.params)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        sample_weight_eval_set: tuple | None = None,
        **kwargs: Any,
    ) -> None:
        # Auto-compute classification sample weights when not provided (legacy behavior)
        if self.task == "classification" and sample_weight is None:
            from validation.utils import calculate_sample_weights  # noqa: PLC0415
            sample_weight = calculate_sample_weights(y_train)

        # Only carve out val split when early stopping is configured
        use_eval_set = bool(self.head_config.params.get("early_stopping_rounds"))
        if use_eval_set and (x_val is None or y_val is None):
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=0.1, random_state=self._random_seed,
            )

        fit_params: dict[str, Any] = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        if sample_weight_eval_set is not None and use_eval_set:
            fit_params["sample_weight_eval_set"] = sample_weight_eval_set

        self._model = self._make_xgb_model()

        if self.device and self.device != "cpu":
            x_train_t = torch.from_numpy(x_train).to(self.device)
            y_train_t = torch.from_numpy(y_train).to(self.device)
            if "sample_weight" in fit_params:
                fit_params["sample_weight"] = torch.from_numpy(fit_params["sample_weight"]).to(self.device)
            if use_eval_set and x_val is not None and y_val is not None:
                x_val_t = torch.from_numpy(x_val).to(self.device)
                y_val_t = torch.from_numpy(y_val).to(self.device)
                if "sample_weight_eval_set" in fit_params:
                    fit_params["sample_weight_eval_set"] = tuple(
                        torch.from_numpy(w).to(self.device) for w in fit_params["sample_weight_eval_set"]
                    )
                self._model.fit(x_train_t, y_train_t, eval_set=[(x_val_t, y_val_t)], **fit_params)
            else:
                self._model.fit(x_train_t, y_train_t, **fit_params)
        else:
            if use_eval_set and x_val is not None and y_val is not None:
                self._model.fit(x_train, y_train, eval_set=[(x_val, y_val)], **fit_params)
            else:
                self._model.fit(x_train, y_train, **fit_params)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XGBoostHead.predict called before fit")
        if self.device and self.device != "cpu":
            return np.asarray(self._model.predict(torch.from_numpy(x_test).to(self.device)))
        return self._model.predict(x_test)

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray | None:
        if self.task != "classification":
            return None
        if self._model is None:
            raise RuntimeError("XGBoostHead.predict_proba called before fit")
        if self.device and self.device != "cpu":
            return np.asarray(self._model.predict_proba(torch.from_numpy(x_test).to(self.device)))
        return self._model.predict_proba(x_test)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("XGBoostHead.save called before fit")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(path)
