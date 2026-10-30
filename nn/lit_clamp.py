import numpy as np
import torch
import torch_geometric
from clamp import CLAMP

from nn._register_model import register_model_type
from nn.multimodal_module import MultiModalModule


class WeightsOnlyFalse:
    def __enter__(self):
        """Enters the context in which torch.load is overridden to force weights_only=False."""
        self.original_torch_load = torch.load

        # Define a wrapper that forces weights_only=False
        def wrapped_load(*args, **kwargs):
            kwargs["weights_only"] = False  # Override any caller-provided value
            return self.original_torch_load(*args, **kwargs)

        torch.load = wrapped_load

    def __exit__(self, exc_type, exc_value, traceback):
        torch.load = self.original_torch_load


@register_model_type("clamp")
class LightningCLAMP(MultiModalModule, modes=("embedding",)):
    def __init__(self, mode: str = "embedding") -> None:
        super().__init__(mode)
        with (
            torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype]),
            WeightsOnlyFalse(),
        ):
            self.clamp = CLAMP(device="cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        return self.clamp.compound_encoder(data.smiles)
