"""Autoencoder AE model."""

import lightning as L  # noqa: N812
import torch
from torch import nn
from torch_geometric.data import Data

from nn._register_model import register_model_type
from utils.minor import MeanOneMinusCosineSimilarity

# TODO: merge AE and LightningAE


class AE(nn.Module):  # Note: this is AE actually :D
    """Autoencoder (AE) model.

    This model is a deep neural network that is used to learn a compressed,
    distributed representation of a dataset. It consists of an encoder and a
    decoder. The encoder maps the input data to a latent space, and the
    decoder maps the latent space back to the original data.

    Parameters
    ----------
    input_dim : int
        Input dimension of the model.
    hidden_dimensions : list[int]
        List of hidden dimensions.
    latent_dim : int
        Dimension of the latent space.
    activation : Optional[nn.Module], optional
        Activation function to use. Defaults to nn.ReLU().
    dropout : float, optional
        Dropout rate to use. Defaults to 0.1.
    batch_norm : bool, optional
        Whether to use batch normalization. Defaults to True.

    Attributes
    ----------
    encoder : nn.Module
        Encoder module.
    decoder : nn.Module
        Decoder module.
    similarity_metric : nn.Module
        Similarity metric module.

    """

    def __init__(  # noqa: PLR0913 # TODO: add config to reduce argument count
        self,
        input_dim: int,
        hidden_dimensions: list[int],
        latent_dim: int,
        activation: nn.Module | None = None,
        dropout: float = 0.1,
        *,
        batch_norm: bool = True,
    ) -> None:
        """Initializes the AE model.

        Parameters
        ----------
        input_dim : int
            Input dimension of the model.
        hidden_dimensions : list[int]
            List of hidden dimensions.
        latent_dim : int
            Dimension of the latent space.
        activation : Optional[nn.Module], optional
            Activation function to use. Defaults to nn.ReLU().
        dropout : float, optional
            Dropout rate to use. Defaults to 0.1.
        batch_norm : bool, optional
            Whether to use batch normalization. Defaults to True.

        """
        if activation is None:
            activation = nn.ReLU()
        super().__init__()

        input_dims_enc = [input_dim, *hidden_dimensions[:-1]]
        output_dims_enc = hidden_dimensions

        input_dims_dec = ([latent_dim] + hidden_dimensions[len(hidden_dimensions) - 1 : 0, :-1],)
        output_dims_dec = hidden_dimensions[len(hidden_dimensions) - 1 : -1, :-1]

        self.encoder = nn.Sequential(
            *(
                nn.Sequential(
                    [nn.Dropout(dropout)] if dropout > 0 else [],
                    (
                        ([nn.BatchNorm1d(input_dim)] if batch_norm else [])
                        + [nn.Linear(input_dim, output_dim), activation]
                    ),
                )
                for input_dim, output_dim in zip(input_dims_enc, output_dims_enc, strict=False)
            ),
            nn.Linear(hidden_dimensions[-1], latent_dim),
        )

        self.decoder = nn.Sequential(
            *(
                nn.Sequential(
                    [nn.Dropout(dropout)] if dropout > 0 else [],
                    (
                        ([nn.BatchNorm1d(input_dim)] if batch_norm else [])
                        + [nn.Linear(input_dim, output_dim), activation]
                    ),
                )
                for input_dim, output_dim in zip(
                    input_dims_dec,
                    output_dims_dec,
                    strict=False,
                )
            ),
            nn.Linear(hidden_dimensions[0], input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input data `x` into a latent space.

        Args:
            x (torch.Tensor): The input data to be encoded.

        Returns:
            torch.Tensor: The encoded input data in the latent space.

        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent space vector `z` back into the original space.

        Args:
            z (torch.Tensor): The latent space vector to be decoded.

        Returns:
            torch.Tensor: The decoded latent space vector in the original space.

        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the AE model.

        Parameters
        ----------
        x : torch.Tensor
            The input data to be encoded and decoded.

        Returns
        -------
        torch.Tensor
            The decoded input data in the original space.

        """
        latent = self.encode(x)
        return self.decode(latent)


@register_model_type("ae")
class LightningAE(L.LightningModule):
    """A PyTorch Lightning Module for an AutoEncoder (AE) model.

    The AE model is a neural network that is trained to copy its input to its output.
    It consists of an encoder and a decoder. The encoder maps the input data to a latent space,
    and the decoder maps the latent space back to the original space.

    The AE model is often used for dimensionality reduction, anomaly detection, and generative
    modeling.

    Parameters
    ----------
    input_dim : int
        The number of features in the input data.
    hidden_dimensions : list[int]
        The dimensions of the hidden layers in the encoder and decoder.
    latent_dim : int
        The dimension of the latent space.
    activation : Optional[nn.Module], optional
        The activation function to use in the encoder and decoder. Defaults to None.
    dropout : float, optional
        The dropout rate to use in the encoder and decoder. Defaults to 0.1.
    batch_norm : bool, optional
        Whether to use batch normalization in the encoder and decoder. Defaults to True.
    loss : str, optional
        The loss function to use. Defaults to "l1".

    Attributes
    ----------
    encoder : nn.Sequential
        The encoder neural network.
    decoder : nn.Sequential
        The decoder neural network.

    """

    def __init__(  # noqa: PLR0913 # TODO: add config to reduce argument count
        self,
        input_dim: int,
        hidden_dimensions: list[int],
        latent_dim: int,
        activation: nn.Module | None = None,
        dropout: float = 0.1,
        *,
        batch_norm: bool = True,
        loss: str = "l1",
    ) -> None:
        """Initializes a LightningAE model.

        Parameters
        ----------
        input_dim : int
            Input dimension of the model.
        hidden_dimensions : list[int]
            List of hidden dimensions.
        latent_dim : int
            Dimension of the latent space.
        activation : Optional[nn.Module], optional
            Activation function to use. Defaults to None.
        dropout : float, optional
            Dropout rate to use. Defaults to 0.1.
        batch_norm : bool, optional
            Whether to use batch normalization. Defaults to True.
        loss : str, optional
            Loss function to use. Can be "l1", "mae", "l2", "mse", or "cosine". Defaults to "l1".

        Raises
        ------
        ValueError
            If the loss is not recognized.

        """
        super().__init__()
        self.AE = AE(
            input_dim=input_dim,
            hidden_dimensions=hidden_dimensions,
            latent_dim=latent_dim,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        loss = loss.lower()
        if loss in {"l1", "mae"}:
            self.loss = nn.L1Loss()
        elif loss in {"l2", "mse"}:
            self.loss = nn.MSELoss()
        elif loss in {"cosine", "cos"}:
            self.loss = MeanOneMinusCosineSimilarity()
        else:
            raise ValueError(loss)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input data `x` into the latent space.

        Parameters
        ----------
        x : torch.Tensor
            The input data to be encoded.

        Returns
        -------
        torch.Tensor
            The encoded input data in the latent space.

        """
        return self.AE.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent space vector `z` back into the original space.

        Parameters
        ----------
        z : torch.Tensor
            The latent space vector to be decoded.

        Returns
        -------
        torch.Tensor
            The decoded latent space vector in the original space.

        """
        return self.AE.decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the AE model.

        Parameters
        ----------
        x : torch.Tensor
            The input data to be encoded and decoded.

        Returns
        -------
        torch.Tensor
            The decoded input data in the original space.

        """
        return self.AE(x)

    def training_step(self, batch: Data, _: int) -> dict[str, torch.Tensor]:
        """Performs a training step.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            The batch of data to use for training.
        _ : int
            Unused argument (batch index).

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the training loss.

        """
        x = batch.y
        x_reconstructed = self.forward(x)
        loss = self.loss(x_reconstructed, x)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch: Data, _: int) -> dict[str, torch.Tensor]:
        """Performs a validation step.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            The batch of data to use for validation.
        _ : int
            Unused argument (batch index).

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the validation loss.

        """
        x = batch
        x_reconstructed = self.forward(x)
        loss = self.loss(x_reconstructed, x)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """Configures the optimizer and learning rate scheduler.

        Returns a dictionary with a single entry, where the key is "optimizer"
        and the value is a torch.optim.Optimizer.

        The optimizer is a Adam optimizer with a learning rate of 3e-4.
        """
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=3e-4,
        )
