import torch
import torch_geometric
from torch import nn

from nn._register_model import register_model_type
from nn.multimodal_module import MultiModalModule


@register_model_type("mean_embeds")
class MeanEmbeds(MultiModalModule, modes=("embedding",)):
    """MeanEmbeds is a simple PyTorch Lightning module that uses the mean of the embeddings as the output.

    Parameters
    ----------
    mode : str
        Model mode.
    n_features : int, optional
        Number of features, by default 240.
    vocab_size : int, optional
        Vocabulary size, by default 19.
    one_hot : bool, optional
        Whether to use one-hot encoding, by default True.

    Attributes
    ----------
    embedding : nn.Embedding
        Embedding layer.
    n_features : int
        Number of features.
    vocab_size : int
        Vocabulary size.
    one_hot : bool
        Whether to use one-hot encoding.

    """

    def __init__(
        self,
        mode: str = "embedding",
        n_features: int = 240,
        vocab_size: int = 19,
        *,
        one_hot: bool = False,
    ) -> None:
        """Initializes the MeanEmbeds module.

        Parameters
        ----------
        mode : str
            Model mode.
        n_features : int, optional
            Number of features, by default 240.
        vocab_size : int, optional
            Vocabulary size, by default 19.
        one_hot : bool, optional
            Whether to use one hot encoding, by default True.

        """
        super().__init__(mode)
        self.aggr = torch_geometric.nn.MeanAggregation()
        if not one_hot:
            self.embedding = nn.Embedding(vocab_size, n_features)
        self.n_features = n_features
        self.vocab_size = vocab_size
        self.one_hot = one_hot

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """Forward pass through the MeanEmbeds module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the MeanEmbeds module.

        """
        x = data.x
        if self.one_hot:
            x = torch.nn.functional.one_hot(x, num_classes=self.vocab_size)
        else:
            x = self.embedding(x)
        x = self.aggr(x, data.batch)
        return x  # noqa: RET504
