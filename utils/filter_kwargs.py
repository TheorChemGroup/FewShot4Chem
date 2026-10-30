"""Provides a helper function to filter kwargs based on the signature of a function."""

import inspect
from collections.abc import Callable
from typing import Any


def filter_kwargs(func: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter out kwargs that are not accepted by func.

    Args:
        func: A callable.
        kwargs: A dictionary of keyword arguments.

    Returns:
        A dictionary of keyword arguments that are accepted by func.

    """
    sig = inspect.signature(func)
    has_var_keyword = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )
    if has_var_keyword:
        return kwargs.copy()
    allowed_names = [
        param.name
        for param in sig.parameters.values()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    return {k: v for k, v in kwargs.items() if k in allowed_names}
