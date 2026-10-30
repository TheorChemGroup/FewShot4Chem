"""Shared SVGP infrastructure for the GP-* heads."""

import logging
import math
import pickle
from pathlib import Path
from typing import Any

import gpytorch
import numpy as np
import torch
from gpytorch.kernels import Kernel
from omegaconf import OmegaConf
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.multioutput import MultiOutputRegressor

from validation.heads.abstract import Head

logger = logging.getLogger(__name__)


class _SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, kernel: Kernel) -> None:
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        var_strat = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            var_dist,
            learn_inducing_locations=True,
        )
        super().__init__(var_strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def _kmeans_inducing(x: np.ndarray, m: int, seed: int) -> np.ndarray:
    if m >= x.shape[0]:
        return x.copy()
    km = KMeans(n_clusters=m, random_state=seed, n_init=10)
    km.fit(x)
    return km.cluster_centers_.astype(x.dtype)


def _median_sq_distance(x: np.ndarray, seed: int, max_rows: int = 1000) -> float:
    """Median pairwise squared Euclidean distance over at most `max_rows` rows (seeded subsample)."""
    if x.shape[0] > max_rows:
        x = x[np.random.default_rng(seed).choice(x.shape[0], size=max_rows, replace=False)]
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
    d2 = torch.cdist(xt, xt).pow(2)
    iu = torch.triu_indices(d2.shape[0], d2.shape[0], offset=1)
    return float(d2[iu[0], iu[1]].median()) if iu.shape[1] else 0.0


def _init_lengthscale(kernel: Kernel, x: np.ndarray, seed: int) -> float | None:
    """Median heuristic: start the lengthscale at sqrt(median pairwise squared distance).

    gpytorch starts every lengthscale at softplus(0) = 0.693. On StandardScaled embeddings the median
    squared distance is ~2*width (1,504 for 768-d MoLFormer, 6,175 for 3,072-d gemini), so the kernel
    is exp(-1500) = 0 everywhere AND so is its gradient: the lengthscale never moves (measured: it
    ended at exactly 0.693), the posterior is the prior, every prediction is the same constant and
    ROC-AUC is exactly 0.500. Kernels without a lengthscale (Tanimoto) are left alone.
    """
    base = getattr(kernel, "base_kernel", kernel)
    if not getattr(base, "has_lengthscale", False):
        return None
    ls = math.sqrt(_median_sq_distance(x, seed))
    if not math.isfinite(ls) or ls <= 0.0:
        return None
    base.lengthscale = ls
    return ls


class _SVGPSingleTargetHead:
    """Fits one SVGP for a single target column.

    `task` selects the likelihood: GaussianLikelihood for regression (default,
    used by the multi-target adaptor below), BernoulliLikelihood for binary
    classification. Everything else about the SVGP/ELBO training loop is
    likelihood-agnostic.
    """

    def __init__(self, head_config: OmegaConf, kernel_factory, device: str, task: str = "regression") -> None:
        self.head_config = head_config
        self._kernel_factory = kernel_factory
        self.device = device
        self.task = task
        self._gp: _SVGPModel | None = None
        self._likelihood: gpytorch.likelihoods.Likelihood | None = None
        self.y_mu_, self.y_sd_ = 0.0, 1.0          # identity unless the regression target is standardized
        self.lengthscale_init_: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        n_train = x.shape[0]
        m = min(int(self.head_config.num_inducing_cap), n_train // 2)
        m = max(m, 2)
        seed = int(self.head_config.seed)
        torch.manual_seed(seed)

        inducing_np = _kmeans_inducing(x, m, seed)
        inducing = torch.from_numpy(inducing_np).to(self.device, dtype=torch.float32)
        kernel = self._kernel_factory(x.shape[1])
        if str(self.head_config.get("lengthscale_init", "median")) == "median":
            self.lengthscale_init_ = _init_lengthscale(kernel, x, seed)
        kernel = kernel.to(self.device)
        self._gp = _SVGPModel(inducing, kernel).to(self.device)
        if self.task == "classification":
            self._likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(self.device)
        else:
            self._likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        # Regression target standardized for the fit, predictions mapped back. Unscaled, the ConstantMean
        # starts at 0 and moves ~lr per step, so a target centred at -3 (ESOL) is predicted near 0 and
        # R2 goes negative.
        y = np.asarray(y, dtype=np.float64)
        if self.task == "regression" and bool(self.head_config.get("standardize_y", True)):
            sd = float(y.std())
            self.y_mu_, self.y_sd_ = float(y.mean()), (sd if sd > 0 else 1.0)
            y = (y - self.y_mu_) / self.y_sd_

        x_t = torch.from_numpy(x.astype(np.float32)).to(self.device)
        y_t = torch.from_numpy(y.astype(np.float32)).to(self.device)

        self._gp.train()
        self._likelihood.train()
        opt = torch.optim.Adam(
            [{"params": self._gp.parameters()}, {"params": self._likelihood.parameters()}],
            lr=float(self.head_config.lr),
        )
        mll = gpytorch.mlls.VariationalELBO(self._likelihood, self._gp, num_data=n_train)
        bs = int(self.head_config.batch_size)
        for _ in range(int(self.head_config.n_iter)):
            perm = torch.randperm(n_train, device=self.device)
            for start in range(0, n_train, bs):
                idx = perm[start : start + bs]
                opt.zero_grad()
                output = self._gp(x_t[idx])
                loss = -mll(output, y_t[idx])
                loss.backward()
                opt.step()
                if loss.isnan():
                    msg = (
                        "SVGP training diverged (NaN loss). "
                        "Consider reducing lr or increasing batch_size in head config."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        if self._gp is None or self._likelihood is None:
            msg = "GP not fitted"
            raise RuntimeError(msg)
        self._gp.eval()
        self._likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x_t = torch.from_numpy(x_test.astype(np.float32)).to(self.device)
            preds = self._likelihood(self._gp(x_t))
            return preds.mean.cpu().numpy() * self.y_sd_ + self.y_mu_


class _SklearnGPAdaptor(RegressorMixin, BaseEstimator):
    """Adapts `_SVGPSingleTargetHead` to sklearn estimator API for MultiOutputRegressor.

    Regression-only: multi-target classification is not implemented, so this
    adaptor always trains its inner head with task="regression".

    Inherits RegressorMixin/BaseEstimator (BaseEstimator on the right, per sklearn's own
    convention) purely so sklearn's tag machinery has something to call: newer sklearn
    (>=1.6) requires `__sklearn_tags__` on any estimator MultiOutputRegressor wraps, and a
    plain duck-typed class without it raises `AttributeError: '_SklearnGPAdaptor' object has
    no attribute '__sklearn_tags__'` from inside MultiOutputRegressor.fit(). get_params/set_params
    below are still hand-written (not the __init__-signature-introspection BaseEstimator
    would otherwise provide) and take precedence over the inherited ones.
    """

    def __init__(self, head_config: OmegaConf, kernel_factory, device: str) -> None:
        self.head_config = head_config
        self.kernel_factory = kernel_factory
        self.device = device
        self._inner: _SVGPSingleTargetHead | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_SklearnGPAdaptor":
        self._inner = _SVGPSingleTargetHead(self.head_config, self.kernel_factory, self.device, task="regression")
        self._inner.fit(x, np.asarray(y).ravel())
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._inner is None:
            msg = "adaptor not fitted"
            raise RuntimeError(msg)
        return self._inner.predict(x)

    def get_params(self, deep: bool = True) -> dict:  # noqa: ARG002, FBT001, FBT002
        return {"head_config": self.head_config, "kernel_factory": self.kernel_factory, "device": self.device}

    def set_params(self, **params: Any) -> "_SklearnGPAdaptor":
        for k, v in params.items():
            setattr(self, k, v)
        return self


class _GPHeadBase(Head):
    """Base class shared by GPHeadRBF, GPHeadMatern, GPHeadTanimoto."""

    KERNEL_NAME: str = ""

    # SVGP on <10k train fits in <10 min on CPU and saves a GPU slot. QM9-scale (80k)
    # would take hours on CPU, so stay on GPU there. Threshold checked at fit time.
    _GPU_TRAIN_SIZE_THRESHOLD = 10_000

    def __init__(self, head_config: OmegaConf) -> None:
        self.head_config = head_config
        self.task = head_config.objective_type
        self._device_override = head_config.get("device", None)
        self.device = "cpu"
        self._gp: _SVGPModel | None = None
        self._likelihood: gpytorch.likelihoods.Likelihood | None = None
        self._multi_wrap: MultiOutputRegressor | None = None
        self._y_mu, self._y_sd = 0.0, 1.0   # single-target regression: standardization undone in predict

    def _select_device(self, n_train: int) -> str:
        if self._device_override is not None:
            return str(self._device_override)
        if torch.cuda.is_available() and n_train >= self._GPU_TRAIN_SIZE_THRESHOLD:
            return "cuda"
        return "cpu"

    def _kernel_factory(self, in_dim: int) -> Kernel:
        raise NotImplementedError

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
                "%s does not support sample_weight; ignoring. "
                "Classification metrics on imbalanced datasets may be biased.",
                type(self).__name__,
            )
        y_arr = np.asarray(y_train)
        self.device = self._select_device(x_train.shape[0])
        is_multi_target = y_arr.ndim == 2 and y_arr.shape[1] > 1
        if self.task == "classification":
            if is_multi_target:
                msg = "GP classification heads support single-target only (not wired through MultiOutputRegressor)"
                raise NotImplementedError(msg)
            single = _SVGPSingleTargetHead(self.head_config, self._kernel_factory, self.device, task="classification")
            single.fit(x_train, y_arr.ravel())
            self._gp = single._gp  # noqa: SLF001
            self._likelihood = single._likelihood  # noqa: SLF001
            return
        if is_multi_target:
            base = _SklearnGPAdaptor(self.head_config, self._kernel_factory, self.device)
            self._multi_wrap = MultiOutputRegressor(base)
            self._multi_wrap.fit(x_train, y_arr)
            return
        single = _SVGPSingleTargetHead(self.head_config, self._kernel_factory, self.device, task="regression")
        single.fit(x_train, y_arr.ravel())
        self._gp = single._gp  # noqa: SLF001
        self._likelihood = single._likelihood  # noqa: SLF001
        self._y_mu, self._y_sd = single.y_mu_, single.y_sd_

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        if self._multi_wrap is not None:
            return self._multi_wrap.predict(x_test)
        if self._gp is None or self._likelihood is None:
            msg = "GP head not fitted"
            raise RuntimeError(msg)
        self._gp.eval()
        self._likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x_t = torch.from_numpy(x_test.astype(np.float32)).to(self.device)
            preds = self._likelihood(self._gp(x_t))
            if self.task == "classification":
                return (preds.mean.cpu().numpy() > 0.5).astype(np.int64)  # noqa: PLR2004
            return preds.mean.cpu().numpy() * self._y_sd + self._y_mu

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray | None:
        if self.task != "classification":
            return None
        if self._gp is None or self._likelihood is None:
            msg = "GP head not fitted"
            raise RuntimeError(msg)
        self._gp.eval()
        self._likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x_t = torch.from_numpy(x_test.astype(np.float32)).to(self.device)
            preds = self._likelihood(self._gp(x_t))
            p1 = preds.mean.cpu().numpy()
            return np.stack([1 - p1, p1], axis=1)

    def save(self, path: Path) -> None:
        if self._gp is None and self._multi_wrap is None:
            msg = "GP head not fitted"
            raise RuntimeError(msg)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._multi_wrap is not None:
            with path.open("wb") as f:
                pickle.dump(self._multi_wrap, f)
            return
        torch.save(
            {
                "gp_state_dict": self._gp.state_dict(),
                "likelihood_state_dict": self._likelihood.state_dict(),
                "kernel_name": self.KERNEL_NAME,
                "y_mu": self._y_mu,
                "y_sd": self._y_sd,
            },
            path,
        )
