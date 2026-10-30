"""Graph Neural networks used in the work are kept here."""

import logging
from collections.abc import Iterable, Sequence
from typing import Literal, NamedTuple

import lightning as L  # noqa: N812
import numpy as np
import torch
import torch_geometric
from lightning.pytorch.utilities import grad_norm
from torch import nn

from nn._register_model import register_model_type
from nn.multimodal_module import UnknownModelModeError
from utils.minor import MeanOneMinusCosineSimilarity

logger = logging.getLogger(__name__)


class PositiveNegative(NamedTuple):
    """Tuple of positive and negative references."""

    positive: torch.Tensor
    negative: torch.Tensor


INVALID_SKIP_CONNECTION = (
    "skip_connection must be a bool or a tuple of tuples of length 2, but got {}"
)

BACKWARD_SKIP_CONNECTION = (
    "cannot skip connection from layer {start} to layer {end}, "
    "because layer {end} is before layer {start}"
)

# TODO: merge MultilayerGATv2 and LightningMultilayerGATv2

# TODO: add other GNN types


class MultilayerGATv2(nn.Module):
    """Multilayer GATv2 model.

    This class implements a graph attention network v2 (GATv2) with multiple
    layers. The model is based on the paper "Graph Attention Networks" by
    Veli koi et al. (2018) and the implementation is based on the code by
    PyTorch Geometric authors.

    Attributes:
        vocab_size: The size of the vocabulary.
        hidden_dim: The dimension of the hidden layers.
        very_hidden_dim: The dimension of the very hidden layers.
        dropout_prob: The probability of dropout.
        heads: The number of attention heads.
        layers: The number of layers.
        skip_connection: The skip connection to use.
        normalize: Whether to normalize the output.

    """

    layers: Iterable[torch_geometric.nn.conv.MessagePassing]

    def __init__(  # noqa: PLR0913
        self,
        vocab_size: int,
        hidden_dim: int,
        *,
        very_hidden_dim: int | None = None,
        edge_dim: int | None = None,
        dropout_prob: float = 0.3,
        heads: int = 10,
        layers: int = 5,
        skip_connection: bool | Sequence[Sequence[int]] = True,
        normalize: bool = False,
        final_shift: bool = False,
        mode: Literal["embedding", "mlm"] = "embedding",
    ) -> None:
        """Initializes the MultilayerGATv2 module.

        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary.
        hidden_dim : int
            Dimension of the hidden layers.
        very_hidden_dim : Optional[int], optional
            Dimension of the very hidden layers, by default None.
        edge_dim : Optional[int], optional
            Dimension of the edge features, by default None.
        dropout_prob : float, optional
            Probability of dropout, by default 0.3.
        heads : int, optional
            Number of attention heads, by default 10.
        layers : int, optional
            Number of layers, by default 5.
        skip_connection : bool or Sequence[Sequence[int]] , optional
            if True, use skip connections from each layer to the next (if they have the same shape)
            if Sequence[Sequence[int]], this should be the sequence of pairs,
            where the first element is the layer from which the skip connection comes
            (starting from 0 - the input tensor),
            and the second element is the layer to which the skip connection goes
            I.e, if (2, 3) is in the sequence, then the **input** (!)
            of layer 2 will be added to the output of layer 3
        final_shift : bool, optional
            if True, apply a final shift to the output of the last layer
            by  adding learnable bias.
            by default False
        normalize : bool, optional
            if True, normalize the output of the model
            by default False
        mode : Literal["embedding", "mlm"], optional
            mode of the model, by default "embedding"


        """
        super().__init__()

        assert 0 <= dropout_prob < 1, "dropout_prob must be between 0 and 1"  # noqa: S101
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"  # noqa: S101

        if very_hidden_dim is None:
            very_hidden_dim = hidden_dim
        else:
            assert very_hidden_dim % heads == 0, "very_hidden_dim must be divisible by heads"  # noqa: S101

        self.embs = nn.Embedding(vocab_size, hidden_dim)

        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.GELU()

        self.layers = nn.ModuleList(
            [
                torch_geometric.nn.GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=very_hidden_dim // heads,
                    heads=heads,
                    edge_dim=edge_dim,
                ),
            ]
            + [
                torch_geometric.nn.GATv2Conv(
                    in_channels=very_hidden_dim,
                    out_channels=very_hidden_dim // heads,
                    heads=heads,
                    edge_dim=edge_dim,
                )
                for _ in range(layers - 2)
            ]
            + [
                torch_geometric.nn.GATv2Conv(
                    in_channels=very_hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    edge_dim=edge_dim,
                ),
            ],
        )

        self.norms = nn.ModuleList(
            [torch_geometric.nn.GraphNorm(in_channels=hidden_dim)]
            + [
                torch_geometric.nn.GraphNorm(in_channels=very_hidden_dim) for _ in range(layers - 1)
            ],
        )

        self.aggr = torch_geometric.nn.MeanAggregation()

        self.proj = nn.Linear(hidden_dim, hidden_dim)
        if isinstance(skip_connection, bool):
            self.skip_connection = skip_connection
        else:
            self.skip_connection = {}
            for pair in skip_connection:
                if len(pair) != 2:  # noqa: PLR2004
                    raise ValueError(
                        INVALID_SKIP_CONNECTION.format(skip_connection),
                    )
                if pair[0] > pair[1]:
                    raise ValueError(
                        BACKWARD_SKIP_CONNECTION.format(start=pair[0], end=pair[1]),
                    )
                self.skip_connection[pair[1]] = pair[0]

        if final_shift:
            self.shift_vector = nn.Parameter(
                torch.rand(hidden_dim) * np.sqrt(hidden_dim),
            )

        self.final_shift = final_shift

        self.normalize = normalize
        self.mlm_predictor = None
        # Note: it's better to assign the mode at the very end,
        # so mlm_predictor may be properly initialized
        self.mode = mode

    @property
    def mode(self) -> Literal["embedding", "mlm"]:
        """Mode of the model (embedding/MLM)."""
        return self._mode

    @mode.setter
    def mode(self, mode: Literal["embedding", "mlm"]) -> None:
        """Sets the mode of the model (embedding/MLM)."""
        mode = mode.lower()
        if mode not in {"embedding", "mlm"}:
            raise UnknownModelModeError(mode)
        if mode == "mlm" and self.mlm_predictor is None:
            self.mlm_predictor = torch_geometric.nn.GATv2Conv(
                in_channels=self.layers[-1].out_channels * self.layers[-1].heads,
                out_channels=self.embs.num_embeddings,
                edge_dim=self.layers[-1].edge_dim,
                heads=1,
            )
        self._mode = mode

    def forward(self, data: torch_geometric.data.Data | torch_geometric.data.Batch) -> torch.Tensor:
        """Forward pass through the GNN.

        Parameters
        ----------
        data : torch_geometric.data.Data or torch_geometric.data.Batch
            Input data containing node features and edge indices.
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Edge indices.
        edge_attr : torch.Tensor, optional
            Edge attributes.

        Returns
        -------
        torch.Tensor
            Output of the GNN.

        """
        x = self.embs(data.x)
        outs = []
        for layer_num, (norm, layer) in enumerate(zip(self.norms, self.layers, strict=False)):
            outs.append(x)
            x = self.dropout(x)
            x = norm.forward(x)
            x = layer.forward(x, edge_index=data.edge_index, edge_attr=data.get("edge_attr", None))
            x = self.activation(x)
            if self.skip_connection:
                if isinstance(self.skip_connection, dict) and layer_num in self.skip_connection:
                    x = x + outs[self.skip_connection[layer_num]]
                if isinstance(self.skip_connection, bool) and x.shape == outs[-1].shape:
                    x = x + outs[-1]
        if self.mode == "mlm":
            return self.mlm_predictor(
                x,
                edge_index=data.edge_index,
                edge_attr=data.get("edge_attr", None),
            )
        x = self.aggr(x, data.batch)
        x = self.proj(x)
        if self.final_shift:
            x = x + self.shift_vector
        if self.normalize:
            x = nn.functional.normalize(x, dim=-1)
        return x


UNKNOWN_LOSS = "Unknown loss: {}"


@register_model_type("gnn")
class LightningMultilayerGATv2(L.LightningModule):
    """PyTorch Lightning Module implementing a Multilayer GATv2 model.

    Parameters
    ----------
    vocab_size : int
        The size of the vocabulary.
    hidden_dim : int
        The size of the hidden dimension.
    very_hidden_dim : int, optional
        The size of the very hidden dimension. If None, it is defaulted to `hidden_dim`.
    edge_dim : int, optional
        The size of the edge dimension. If None, it is not used.
    dropout_prob : float, default=0.3
        The dropout probability.
    heads : int, default=4
        The number of attention heads.
    layers : int, default=5
        The number of layers.

    Attributes
    ----------
    model : nn.Module
        The underlying PyTorch model.

    """

    def __init__(  # noqa: PLR0913
        self,
        vocab_size: int,
        hidden_dim: int,
        *,
        very_hidden_dim: int | None = None,
        edge_dim: int | None = None,
        dropout_prob: float = 0.3,
        heads: int = 4,
        layers: int = 5,
        loss: str = "l1",
        use_triplet_loss: bool = False,
        use_difference_loss: bool = False,
        difference_loss_weight: float = 0.5,
        margin: float = 1,
        skip_connection: bool = True,
        normalize: bool = False,
        debug: bool = False,
        gamma: float = 0.95,
        final_shift: bool = False,
        mode: Literal["embedding", "mlm"] = "embedding",
    ) -> None:
        """Initializes the LightningMultilayerGATv2 module.

        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary.
        hidden_dim : int
            Dimension of the hidden layers.
        very_hidden_dim : Optional[int], optional
            Dimension of the very hidden layers, by default None.
        edge_dim : Optional[int], optional
            Dimension of the edge features, by default None.
        dropout_prob : float, optional
            Probability of dropout, by default 0.3.
        heads : int, optional
            Number of attention heads, by default 4.
        layers : int, optional
            Number of layers, by default 5.
        loss : str, optional
            Loss function to use, by default "l1".
        use_triplet_loss : bool, optional
            Flag to use triplet loss, by default False.
        use_difference_loss : bool, optional
            Flag to use difference loss, by default False.
        difference_loss_weight : float, optional
            Weight for difference loss, by default 0.5.
        margin : float, optional
            Margin for triplet loss, by default 1.
        skip_connection : bool | Sequence[Sequence[int]], optional
            if True, use skip connections from each layer to the next (if they have the same shape)
            if Sequence[Sequence[int]], this should be the sequence of pairs,
            where the first element is the layer from which the skip connection comes
            (starting from 0 - the input tensor),
            and the second element is the layer to which the skip connection goes
            I.e, if (2, 3) is in the sequence, then the **input** (!)
            of layer 2 will be added to the output of layer 3
        normalize : bool, optional
            Whether to normalize the embeddings on output, by default False.
        final_shift : bool, optional
            whether to apply a final shift by adding a trainable vector, by default False.
        mode : Literal["embedding", "mlm"], optional
            Mode for the model, by default "embedding".
        debug : bool, optional
            Flag for debugging, by default False.
        gamma : float, optional
            Gamma for the learning rate scheduler, by default 0.95.


        """
        super().__init__()
        self.model = MultilayerGATv2(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            very_hidden_dim=very_hidden_dim,
            dropout_prob=dropout_prob,
            heads=heads,
            layers=layers,
            skip_connection=skip_connection,
            normalize=normalize,
            final_shift=final_shift,
            mode=mode,
            edge_dim=edge_dim,
        )
        self.loss_name = loss
        if loss.lower() == "l1" or loss.lower() == "mae":
            self.loss = nn.L1Loss()
        elif loss.lower() == "l2" or loss.lower() == "mse":
            self.loss = nn.MSELoss()
        elif loss.lower() == "cosine" or loss.lower() == "cos":
            self.loss = MeanOneMinusCosineSimilarity()
        elif loss.lower() == "nll":
            self.loss = nn.NLLLoss()
        elif loss.lower() == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(UNKNOWN_LOSS.format(loss))
        self.use_triplet_loss = use_triplet_loss
        self.use_difference_loss = use_difference_loss
        self.difference_loss_weight = difference_loss_weight
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        self.normalize = normalize
        self.example_input_array = torch_geometric.data.Batch.from_data_list(
            [
                torch_geometric.data.Data(
                    x=torch.zeros((vocab_size,), dtype=torch.int64),
                    edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
                    y=torch.zeros((hidden_dim,)),
                    negative=torch.zeros((hidden_dim,)),
                    edge_attr=torch.zeros((2, edge_dim), dtype=torch.float32)
                    if edge_dim is not None
                    else None,
                )
                for _ in range(12)
            ],
        )
        self.debug = debug
        self._optimizer = None
        self.lr = 3e-4
        self._gamma = gamma if gamma is not None else 0.95

    def forward(self, batch: torch_geometric.data.Batch | PositiveNegative) -> torch.Tensor:
        """Forward pass through the Lightning module.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Input batch.

        Returns
        -------
        torch.Tensor
            Output of the model.

        """
        if isinstance(batch, PositiveNegative):
            positive = self.model.forward(batch.positive)
            negative = self.model.forward(batch.negative)
            return PositiveNegative(positive, negative)
        return self.model.forward(batch)

    def preproccess(
        self,
        batch: torch_geometric.data.Batch | PositiveNegative,
        out: torch.Tensor,
    ) -> torch_geometric.data.Batch:
        """Preprocess the batch before computing loss.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Input batch.
        out : torch.Tensor
            Model output.

        Returns
        -------
        torch_geometric.data.Batch
            Preprocessed batch.

        """
        if isinstance(batch, PositiveNegative):
            batch.positive.y = batch.y.view_as(out[0])
            if self.normalize:
                batch.positive.y = nn.functional.normalize(batch.positive.y, dim=1)
                batch.negative.y = nn.functional.normalize(batch.negative.y, dim=1)
        elif self.mode == "embedding":
            batch.y = batch.y.view_as(out)
            if self.normalize:
                batch.y = nn.functional.normalize(batch.y, dim=1)

        return batch

    def get_loss(self, out: torch.Tensor, batch: torch_geometric.data.Batch) -> torch.Tensor:
        """Compute the loss.

        Parameters
        ----------
        out : torch.Tensor
            Model output.
        batch : torch_geometric.data.Batch
            Input batch.

        Returns
        -------
        torch.Tensor
            Computed loss.

        """
        if self.mode == "mlm":
            if isinstance(self.loss, (nn.NLLLoss, nn.CrossEntropyLoss)):
                loss = self.loss(out, batch.x_unmasked)
            else:
                onehot = torch.nn.functional.one_hot(batch.x_unmasked, num_classes=out.shape[1])
                loss = self.loss(out, onehot.float())
        else:
            if self.use_triplet_loss:
                loss = self.triplet_loss(
                    out.positive,
                    batch.positive.y.view_as(out.positive),
                    out.negative.view_as(out.positive),
                )
            else:
                loss = self.loss(out, batch.y.view_as(out))
            if self.use_difference_loss:
                true_difference = batch.y[:-1] - batch.y[1:]
                difference = out[:-1] - out[1:]
                loss = loss * (
                    1 - self.difference_loss_weight
                ) + self.difference_loss_weight * self.loss(difference, true_difference)
        return loss

    def training_step(self, batch: torch_geometric.data.Batch, _: int) -> torch.Tensor:
        """Perform a training step.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Input batch.
        _ : int
            Placeholder for unused arguments.

        Returns
        -------
        torch.Tensor
            Training loss.

        """
        out = self.forward(batch)
        batch = self.preproccess(batch, out)
        loss = self.get_loss(out, batch)
        self.log("train_loss", loss, batch_size=32)
        return loss

    def validation_step(self, batch: torch_geometric.data.Batch, _: int) -> torch.Tensor:
        """Perform a validation step.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Input batch.
        _ : Any
            Placeholder for unused arguments.

        Returns
        -------
        torch.Tensor
            Validation loss.

        """
        out = self.forward(batch)
        batch = self.preproccess(batch, out)
        loss = self.get_loss(out, batch)

        self.log("val_loss", loss, batch_size=32)
        return loss

    def configure_optimizers(
        self,
    ) -> dict[str, torch.optim.Optimizer | torch.optim.lr_scheduler._LRScheduler]:
        """Configures the optimizer and learning rate scheduler."""
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": self._optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ExponentialLR(
                self._optimizer,
                gamma=self._gamma,
            ),
        }

    def on_before_optimizer_step(self, _: torch.optim.Optimizer) -> None:
        """Hook that is called before the optimizer step.

        Currently, it logs the 2-norm of the gradients for each layer if debug mode is enabled.

        Args:
            _ (torch.optim.Optimizer): The optimizer (unused).

        """
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        if self.debug:
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)

    @property
    def mode(self) -> Literal["embedding", "mlm"]:
        """Mode of the model (embedding/MLM)."""
        return self.model.mode

    @mode.setter
    def mode(self, value: Literal["embedding", "mlm"]) -> None:
        """Set the mode of the model (embedding/MLM)."""
        self.model.mode = value
