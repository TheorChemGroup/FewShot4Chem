"""Provide additional typing hints."""

from collections.abc import Callable
from typing import Any, Protocol

import torch
import torch_geometric


class HasFitAndTransform(Protocol):
    """Protocol for models that have fit and transform methods."""

    fit: Callable[..., Any]
    transform: Callable[..., Any]


class HasFitTransform(HasFitAndTransform):
    """Protocol for models that have fit, transform and fit_transform methods."""

    fit_transform: Callable[..., Any]


DataItem = dict[str, torch.Tensor] | torch_geometric.data.Data
