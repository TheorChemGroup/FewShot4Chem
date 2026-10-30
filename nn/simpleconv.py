import torch
import torch_geometric
from torch import nn
from torch.nn import init

from nn._register_model import register_model_type
from nn.multimodal_module import MultiModalModule


@register_model_type("simple_conv")
class LightningSimpleConv(MultiModalModule, modes=("embedding",)):
    def __init__(
        self,
        mode: str = "embedding",
        layers: int = 1,
        n_features: int = 240,
        vocab_size: int = 19,
        *,
        orthogonal_init: bool = False,
        one_hot: bool = False,
    ) -> None:
        """Initializes the LightningSimpleConv module.

        Parameters
        ----------
        mode : str, optional
            Model mode, by default "embedding".
        n_features : int, optional
            Number of features, by default 240.
        vocab_size : int, optional
            Vocabulary size, by default 19.
        orthogonal_init : bool, optional
            Whether to use orthogonal initialization for the embedding layer, by default False.

        """
        super().__init__(mode=mode)
        if not one_hot:
            self.embedding = nn.Embedding(vocab_size, n_features)
        self.n_features = n_features
        self.vocab_size = vocab_size
        self.simple_conv = torch_geometric.nn.SimpleConv()
        self.aggr = torch_geometric.nn.MeanAggregation()
        self.orthogonal_init = orthogonal_init
        if self.orthogonal_init:
            init.orthogonal_(self.embedding.weight)
        self.n_layers = layers
        self.one_hot = one_hot

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """Forward pass through the LightningSimpleConv module.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data.

        Returns
        -------
        torch.Tensor
            Output of the module.

        """
        x = data.x
        edge_index = data.edge_index

        x = (
            torch.nn.functional.one_hot(x, num_classes=self.vocab_size)
            if self.one_hot
            else self.embedding(x)
        )
        for _ in range(self.n_layers):
            x = self.simple_conv(x, edge_index)
        x = self.aggr(x, data.batch)
        return x  # noqa: RET504

    def reset_parameters(self) -> None:
        """Resets the parameters of the module.

        Calls the parent class's `reset_parameters` method, and if `self.orthogonal_init` is `True`,
        reinitializes the `self.embedding` layer with orthogonal weights.
        """
        super().reset_parameters()
        if not self.one_hot and self.orthogonal_init:
            init.orthogonal_(self.embedding.weight)
