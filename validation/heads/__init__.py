"""Evaluation heads package."""

from validation.heads._register_head import (
    HEAD_REGISTRY,
    load_head_from_config,
    register_head_type,
)
from validation.heads.abstract import Head

# Import all head modules so their @register_head_type decorators fire on package import.
import validation.heads.gp_matern  # noqa: F401, E402
import validation.heads.gp_rbf  # noqa: F401, E402
import validation.heads.gp_tanimoto  # noqa: F401, E402
import validation.heads.linear  # noqa: F401, E402
import validation.heads.mlp  # noqa: F401, E402
import validation.heads.tabicl  # noqa: F401, E402
import validation.heads.tabpfn  # noqa: F401, E402
import validation.heads.xgboost  # noqa: F401, E402

__all__ = ["HEAD_REGISTRY", "Head", "load_head_from_config", "register_head_type"]
