"""GP head with Tanimoto kernel for binary fingerprints (Morgan/ECFP)."""

import gpytorch
import numpy as np
import torch

from validation.heads._register_head import register_head_type
from validation.heads.gp_base import _GPHeadBase

_NON_BINARY_MSG = (
    "GPHeadTanimoto requires binary fingerprint input; got values outside {0, 1}. "
    "Use this head only with the 4 Morgan representations."
)


class _TanimotoKernel(gpytorch.kernels.Kernel):
    """Tanimoto similarity: K(x,y) = <x,y> / (||x||² + ||y||² - <x,y>)."""

    has_lengthscale = False

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,  # noqa: FBT001, FBT002
        last_dim_is_batch: bool = False,  # noqa: FBT001, FBT002, ARG002
        **_: object,
    ) -> torch.Tensor:
        if diag:
            xx = (x1 * x2).sum(-1)
            x1_sq = (x1 * x1).sum(-1)
            x2_sq = (x2 * x2).sum(-1)
            return xx / (x1_sq + x2_sq - xx + 1e-9)
        xx = x1 @ x2.transpose(-1, -2)
        x1_sq = (x1 * x1).sum(-1, keepdim=True)
        x2_sq = (x2 * x2).sum(-1, keepdim=True).transpose(-1, -2)
        return xx / (x1_sq + x2_sq - xx + 1e-9)


def _is_binary(x: np.ndarray) -> bool:
    uniques = np.unique(x)
    return uniques.size <= 2 and set(uniques.tolist()) <= {0.0, 1.0, 0, 1}  # noqa: PLR2004


@register_head_type("gp-tanimoto")
class GPHeadTanimoto(_GPHeadBase):
    KERNEL_NAME = "tanimoto"

    def _kernel_factory(self, in_dim: int):  # noqa: ARG002
        return gpytorch.kernels.ScaleKernel(_TanimotoKernel())

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs: object) -> None:
        if not _is_binary(np.asarray(x_train)):
            raise NotImplementedError(_NON_BINARY_MSG)
        super().fit(x_train, y_train, **kwargs)
