"""Provides a unctionality to calculate fingerprints of molecules."""

import numpy as np
import torch
import torch_geometric
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch import nn

from nn._register_model import register_model_type

INVALID_SMILES = "Invalid SMILES: {}"


@register_model_type("fingerprints")
class FingerprintCreator(nn.Module):
    """FingerprintCreator generates fingerprints of molecules.

    Returns
    -------
    Fingerprints of molecules.

    Notes
    -----
    This module uses the RDKit fingerprint generator to generate fingerprints of molecules.
    The fingerprints are then converted to a numpy array and returned.

    """

    def __init__(  # TODO: add config to reduce argument count
        self,
        *,
        radius: int = 2,
        fpSize: int = 768,
    ) -> None:
        """Initializes a FingerprintCreator object.

        Parameters
        ----------
        radius : int, default=2
            The radius of the fingerprint.
        fpSize : int, default=768
            The size of the fingerprint.

        Returns
        -------
        None

        """
        super().__init__()
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=fpSize,
        )
        self.mode = "embedding"

    def forward(self, data: torch_geometric.data.Data) -> np.array:
        """Computes the Morgan fingerprint of a molecule from its SMILES string.

        Args:
            data (torch_geometric.data.Data): The input data.

        Returns:
            np.array: The computed fingerprints.

        Raises:
            ValueError: If the SMILES string is invalid.

        """
        fps = [None] * len(data.smiles)
        for i, smiles in enumerate(data.smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(INVALID_SMILES.format(smiles))
            fp = self.generator.GetFingerprint(mol)
            fps[i] = fp
        return torch.tensor(fps, device=data.x.device)
