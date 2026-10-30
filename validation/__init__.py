"""Utility functions for validation."""

__all__ = ["select_validator"]


def __getattr__(name):
    if name == "select_validator":
        from validation.select_validator import select_validator
        return select_validator
    raise AttributeError(f"module 'validation' has no attribute {name!r}")
