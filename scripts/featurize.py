"""Compute frozen representations for the small-sample cohorts.

Loads the full dataset with `utils.datasets.load_dataset` (downloaded on first use into
`datasets/`), takes the 100 cohort molecules by their `dataset_index`, and embeds them with the
model named by a config in `models/configs/` through `utils.embed_dataset`. This produces the
same rows the paper cut out of full-dataset embedding caches.

    python scripts/featurize.py --dataset ESOL --model-config models/configs/descriptors/rdkit.yaml

Output: `features/small_sample/<dataset>__<model name>.npy` (cohort order), read by
`scripts/fit_head.py`.
"""

import json
import sys
from pathlib import Path

import click
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COHORTS = ROOT / "data/cohorts/small_sample.json"
OUT = ROOT / "features/small_sample"


def same_molecule(a, b):
    """Equal SMILES, or the same molecule once stereo marks are dropped (the cohort tables may
    write double-bond directions that the dataset omits)."""
    from rdkit import Chem

    return a == b or Chem.CanonSmiles(a, useChiral=0) == Chem.CanonSmiles(b, useChiral=0)


def cohort_subset(dataset_name, cohort, data_root):
    from utils.datasets import load_dataset

    dataset = load_dataset(dataset_name, root=Path(data_root))
    subset = [dataset[int(r["dataset_index"])] for r in cohort["records"]]
    for item, record in zip(subset, cohort["records"], strict=True):
        # The cohort was drawn from a SMILES table of the same dataset; refuse a silent row shift.
        assert same_molecule(item.smiles, record["smiles"]), (record["dataset_index"], item.smiles, record["smiles"])
    return subset


@click.command()
@click.option("--dataset", required=True, help="BACE, Lipophilicity, ESOL, FreeSolv or QM9.")
@click.option("--model-config", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--data-root", default=str(ROOT / "datasets"), show_default=True)
@click.option("--batch-size", type=int, default=None, help="Default: the config's batch_size.")
def main(dataset, model_config, data_root, batch_size):
    from omegaconf import OmegaConf

    from utils.embed_dataset import EmbeddingParams, embed_dataset
    from utils.load_models import load_model

    cfg = OmegaConf.load(model_config)
    cohort = json.loads(COHORTS.read_text())[dataset]
    subset = cohort_subset(dataset, cohort, data_root)
    model = load_model(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{dataset}__{cfg.name}.npy"
    params = EmbeddingParams(
        path=dest,  # embed_dataset writes the array here atomically
        batch_size=batch_size or cfg.get("batch_size", 64),
        return_embeddings=True,
        recalculate=True,
    )
    x = np.asarray(embed_dataset(model, subset, params))
    assert x.shape[0] == 100 and np.isfinite(x).all(), x.shape
    print(f"{dest.relative_to(ROOT)} {x.shape} {x.dtype}")


if __name__ == "__main__":
    main()
