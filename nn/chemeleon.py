"""Wraps CheMeleon (Heid et al. 2025) as a frozen, no-finetuning SMILES embedder.

Vendors the reference implementation from
https://github.com/JacksonBurns/CheMeleon/blob/main/chemeleon_fingerprint.py (the authors'
README explicitly invites copying this file rather than depending on a package — there is no
`chemeleon` package on PyPI). Only its own dependency, `chemprop>=2.2.0`, is a project
dependency. Weights (`chemeleon_mp.pt`, Zenodo record 15460715) are fetched on first
instantiation and cached under `~/.chemprop/`.

The vendored code is Copyright (c) 2025 Jackson Burns, MIT License (see the top-level LICENSE).
"""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
import torch_geometric
from chemprop import featurizers
from chemprop import nn as chemprop_nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from rdkit.Chem import Mol, MolFromSmiles
from torch import nn

from nn._register_model import register_model_type

CHEMELEON_WEIGHTS_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
INVALID_SMILES = "Invalid SMILES: {}"


class CheMeleonFingerprint:
    """Reference CheMeleon fingerprint extractor, vendored verbatim (see module docstring)."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = chemprop_nn.MeanAggregation()
        ckpt_dir = Path.home() / ".chemprop"
        ckpt_dir.mkdir(exist_ok=True)
        mp_path = ckpt_dir / "chemeleon_mp.pt"
        if not mp_path.exists():
            urlretrieve(CHEMELEON_WEIGHTS_URL, mp_path)  # noqa: S310
        chemeleon_mp = torch.load(mp_path, weights_only=True)
        mp = chemprop_nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
        mp.load_state_dict(chemeleon_mp["state_dict"])
        self.output_dim = mp.output_dim
        self.model = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=RegressionFFN(input_dim=mp.output_dim),  # not actually used
        )
        self.model.eval()
        if device is not None:
            self.model.to(device=device)

    def __call__(self, molecules: list[str | Mol]) -> np.ndarray:
        bmg = BatchMolGraph(
            [self.featurizer(MolFromSmiles(m) if isinstance(m, str) else m) for m in molecules],
        )
        bmg.to(device=self.model.device)
        with torch.no_grad():
            return self.model.fingerprint(bmg).numpy(force=True)


@register_model_type("chemeleon")
class CheMeleon(nn.Module):
    """LightningModule-compatible wrapper exposing CheMeleon through the grid's embedding path.

    Frozen foundation-model fingerprint, no fine-tuning — same shape as ``FingerprintCreator``
    (a deterministic feature extractor, not a trainable Lightning module).
    """

    def __init__(self) -> None:
        """Initializes CheMeleon, downloading weights on first use if not already cached."""
        super().__init__()
        self.fingerprint = CheMeleonFingerprint()
        self.mode = "embedding"
        self.output_dim = self.fingerprint.output_dim

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """Computes the CheMeleon fingerprint for a batch of SMILES strings.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data containing a "smiles" field (list of SMILES strings).

        Returns
        -------
        torch.Tensor
            CheMeleon embeddings, shape (batch, self.output_dim).

        """
        for smiles in data.smiles:
            if MolFromSmiles(smiles) is None:
                raise ValueError(INVALID_SMILES.format(smiles))
        embeddings = self.fingerprint(list(data.smiles))
        return torch.tensor(embeddings, dtype=torch.float32,
                            # A record built from SMILES alone (FS-Mol's embed step) has no node
                            # features, so data.x is None and cannot name a device. The output is
                            # moved to CPU by embed_dataset regardless; the main grid always has x
                            # and takes the first branch, exactly as before.
                            device=data.x.device if getattr(data, "x", None) is not None else "cpu")
