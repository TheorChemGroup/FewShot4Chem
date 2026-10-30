"""GP head with Matern kernel."""

import gpytorch

from validation.heads._register_head import register_head_type
from validation.heads.gp_base import _GPHeadBase


@register_head_type("gp-matern")
class GPHeadMatern(_GPHeadBase):
    KERNEL_NAME = "matern"

    def _kernel_factory(self, in_dim: int):  # noqa: ARG002
        nu = float(self.head_config.get("nu", 2.5))
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu))
