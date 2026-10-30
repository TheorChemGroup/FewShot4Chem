"""GP head with RBF (squared-exponential) kernel."""

import gpytorch

from validation.heads._register_head import register_head_type
from validation.heads.gp_base import _GPHeadBase


@register_head_type("gp-rbf")
class GPHeadRBF(_GPHeadBase):
    KERNEL_NAME = "rbf"

    def _kernel_factory(self, in_dim: int):  # noqa: ARG002
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
