"""Registry pattern for evaluation heads. Mirrors nn/_register_model.py."""

from typing import TYPE_CHECKING

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from validation.heads.abstract import Head

HEAD_REGISTRY: dict[str, type] = {}

UNKNOWN_HEAD_TYPE_MSG = (
    "Unknown head_type {name!r}. Registered: {registered}. "
    "Did you forget to import the head module?"
)


def register_head_type(name: str):
    """Register a Head subclass under `name` in HEAD_REGISTRY."""

    def decorator(cls):
        HEAD_REGISTRY[name] = cls
        return cls

    return decorator


def load_head_from_config(head_config: OmegaConf) -> "Head":
    """Instantiate a head from a config with `head_type` field."""
    name = head_config.head_type
    if name not in HEAD_REGISTRY:
        raise KeyError(
            UNKNOWN_HEAD_TYPE_MSG.format(name=name, registered=sorted(HEAD_REGISTRY)),
        )
    return HEAD_REGISTRY[name](head_config=head_config)
