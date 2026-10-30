"""Train/test split strategies for the validation pipeline.

Two strategies:
- "random": plain shuffled split via sklearn.train_test_split (default; matches main grid).
- "scaffold": Bemis-Murcko scaffold split (MoleculeNet/chemprop convention).
  Molecules sharing a Murcko scaffold land in the same partition. Larger
  scaffold buckets bias toward train; smaller buckets are randomized per seed.

Returns train/test integer index arrays, leaving downstream slicing/scaling to
the caller (so the existing prepare_data flow is unchanged apart from how
indices are chosen).
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Buckets >= bucket_cutoff_size go to train; smaller buckets are randomized then
# greedily filled into test/val until test_size is met. The cutoff matches the
# chemprop default for MoleculeNet scaffold-balanced.
_BIG_BUCKET_CUTOFF_FRACTION = 0.5  # of test_size


def _murcko_scaffold(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=False)


def random_split_indices(
    n: int,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)
    return train_idx, test_idx


def scaffold_split_indices(
    smiles_list: list[str],
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(smiles_list)
    scaffolds: dict[str, list[int]] = {}
    n_invalid = 0
    for i, smi in enumerate(smiles_list):
        try:
            scaff = _murcko_scaffold(smi)
        except Exception as exc:  # noqa: BLE001
            logger.warning("scaffold computation failed for idx %d (%s); routing to train", i, exc)
            scaff = ""
            n_invalid += 1
        scaffolds.setdefault(scaff, []).append(i)
    if n_invalid:
        logger.warning("scaffold_split: %d / %d molecules had unparseable SMILES", n_invalid, n)

    test_target = int(round(test_size * n))
    big_cutoff = int(round(_BIG_BUCKET_CUTOFF_FRACTION * test_target))
    big_buckets = [b for b in scaffolds.values() if len(b) >= big_cutoff]
    small_buckets = [b for b in scaffolds.values() if len(b) < big_cutoff]

    rng = np.random.default_rng(seed)
    rng.shuffle(small_buckets)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for bucket in big_buckets:
        train_idx.extend(bucket)
    for bucket in small_buckets:
        if len(test_idx) + len(bucket) <= test_target:
            test_idx.extend(bucket)
        else:
            train_idx.extend(bucket)

    if len(test_idx) == 0:
        msg = (
            f"scaffold_split produced empty test set (n={n}, test_target={test_target}, "
            f"big_buckets={len(big_buckets)}, small_buckets={len(small_buckets)}); "
            "all scaffolds exceeded the big-bucket cutoff — dataset too clustered."
        )
        raise ValueError(msg)
    if len(train_idx) == 0:
        msg = f"scaffold_split produced empty train set (n={n}, test_target={test_target})"
        raise ValueError(msg)

    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def get_split_indices(
    n: int,
    *,
    strategy: str,
    test_size: float,
    seed: int,
    smiles_list: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "random":
        return random_split_indices(n, test_size=test_size, seed=seed)
    if strategy == "scaffold":
        if smiles_list is None or len(smiles_list) != n:
            msg = (
                f"scaffold split requires smiles_list of length {n}; "
                f"got {None if smiles_list is None else len(smiles_list)}"
            )
            raise ValueError(msg)
        return scaffold_split_indices(smiles_list, test_size=test_size, seed=seed)
    msg = f"unknown split strategy: {strategy!r} (expected 'random' or 'scaffold')"
    raise ValueError(msg)
