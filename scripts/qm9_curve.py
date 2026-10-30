"""Figure 2 (QM9 ε_LUMO learning curve) at support 100 and 10,000.

Cohorts (`scripts/make_cohorts.py` documents and re-checks the draws):
* support 10,000: the random 12,500 molecules in `data/cohorts/qm9_random12500_indices.txt`;
  repeat s splits them 9,000 fit / 1,000 validation / 2,500 test;
* support 100: the random 125 (`data/cohorts/qm9_random125.json`), 90 / 10 / 25, nested in the
  12,500.
Support 1,000 (uniform 1,250) needs no wrapper: frozen heads ran with
`runners/run_one_eval.py --protocol uniform`, adaptations with
`runners/finetune_one.py --split grid --regime full-data --lr-schedule none`.

Commands:

    python scripts/qm9_curve.py cohort                      # QM9 rows of the 12,500 -> data/
    python scripts/qm9_curve.py featurize --model-config models/configs/descriptors/rdkit.yaml
    python scripts/qm9_curve.py head --support 10000 --model rdkit-descriptors --head linear
    python scripts/qm9_curve.py finetune --support 10000 --model-config ... [--lora-rank 8]

Frozen heads at both sizes: features standardised on the fit molecules, head seed = repeat, the
linear head fitted without validation data, other heads given the validation set for early
stopping. Adaptations: `finetune_one.py` on the cohort with its own split (test 20%, then 10% of
the rest for validation); support 10,000 uses the full-data regime (lr 5e-6, LoRA 5e-4, 20
epochs, patience 6), support 100 the low-data regime (lr 1e-5, LoRA 5e-4, 30 epochs, patience 8)
with batch size 256. Support-10,000 adaptations ran with batch size 16, except the LoRA runs
listed with batch size 256 in `results/figure2_candidate_repeats.csv`.
"""

import json
import sys
import types
from pathlib import Path

import click
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "runners"), str(Path(__file__).resolve().parent)]

COHORT_12500 = ROOT / "data/cohorts/qm9_random12500.json"
INDICES_12500 = ROOT / "data/cohorts/qm9_random12500_indices.txt"
COHORT_125 = ROOT / "data/cohorts/qm9_random125.json"
FEATURES = ROOT / "features/qm9_random12500"
HEADS = ("linear", "mlp", "xgboost", "gp-rbf", "gp-matern", "tabpfn", "tabicl")
QM9_LUMO_COLUMN = 3  # epsilon_lumo (eV) in the QM9 targets of utils.datasets


def load_rows(support):
    if not COHORT_12500.is_file():
        raise click.ClickException("run `python scripts/qm9_curve.py cohort` first")
    rows = json.loads(COHORT_12500.read_text())["records"]
    if support == 10000:
        return rows, None
    small = json.loads(COHORT_125.read_text())
    rows = [rows[i] for i in small["rows_in_12500"]]
    assert [r["dataset_index"] for r in rows] == [r["dataset_index"] for r in small["records"]]
    return rows, small


def split(support, repeat, small):
    from make_cohorts import regime_split

    if support == 10000:
        return regime_split(12500, repeat)
    s = small["splits"][str(repeat)]
    return tuple(np.asarray(s[k]) for k in ("fit", "val", "test"))


@click.group()
def cli():
    pass


@cli.command()
@click.option("--data-root", default=str(ROOT / "datasets"), show_default=True)
def cohort(data_root):
    """Write the SMILES and ε_LUMO (eV) of the 12,500 QM9 molecules."""
    from featurize import same_molecule
    from utils.datasets import load_dataset

    qm9 = load_dataset("QM9", root=Path(data_root))
    indices = [int(x) for x in INDICES_12500.read_text().split()]
    records = []
    for i in indices:
        d = qm9[i]
        records.append({"dataset_index": i, "smiles": d.smiles,
                        "epsilon_lumo_eV": float(np.asarray(d.y).ravel()[QM9_LUMO_COLUMN])})
    small = json.loads(COHORT_125.read_text())
    for r, s in zip((records[i] for i in small["rows_in_12500"]), small["records"], strict=True):
        assert r["dataset_index"] == s["dataset_index"] and same_molecule(r["smiles"], s["smiles"])
        assert abs(r["epsilon_lumo_eV"] - s["epsilon_lumo_eV"]) < 1e-6, (r, s)
    COHORT_12500.write_text(json.dumps({"records": records}) + "\n")
    print(f"{COHORT_12500.relative_to(ROOT)}: {len(records)} molecules")


@cli.command()
@click.option("--model-config", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--data-root", default=str(ROOT / "datasets"), show_default=True)
def featurize(model_config, data_root):
    """Frozen representation of the 12,500 molecules (rows in cohort order)."""
    from omegaconf import OmegaConf

    from utils.datasets import load_dataset
    from utils.embed_dataset import EmbeddingParams, embed_dataset
    from utils.load_models import load_model

    cfg = OmegaConf.load(model_config)
    qm9 = load_dataset("QM9", root=Path(data_root))
    subset = [qm9[int(i)] for i in INDICES_12500.read_text().split()]
    FEATURES.mkdir(parents=True, exist_ok=True)
    dest = FEATURES / f"{cfg.name}.npy"
    params = EmbeddingParams(path=dest, batch_size=cfg.get("batch_size", 64), recalculate=True)
    x = np.asarray(embed_dataset(load_model(cfg), subset, params))
    print(f"{dest.relative_to(ROOT)} {x.shape} {x.dtype}")


@cli.command()
@click.option("--support", type=click.Choice(["100", "10000"]), required=True)
@click.option("--model", "model_name", required=True)
@click.option("--head", type=click.Choice(HEADS), required=True)
@click.option("--repeats", default="0-4", show_default=True)
@click.option("--out", type=click.Path(dir_okay=False), default=None, help="JSON-lines output.")
def head(support, model_name, head, repeats, out):
    """Fit a frozen-feature head and report validation / test R²."""
    from omegaconf import OmegaConf
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    from fit_head import head_config_path, parse_repeats
    from validation.heads import load_head_from_config

    support = int(support)
    rows, small = load_rows(support)
    x_all = np.load(FEATURES / f"{model_name}.npy", mmap_mode="r")
    if small is not None:
        x_all = x_all[small["rows_in_12500"]]
    x_all = np.asarray(x_all)
    y = np.asarray([r["epsilon_lumo_eV"] for r in rows], dtype=np.float32)
    sink = open(out, "a") if out else None  # noqa: SIM115
    for repeat in parse_repeats(repeats):
        fit, val, test = split(support, repeat, small)
        scaler = StandardScaler().fit(x_all[fit])
        cfg = OmegaConf.load(head_config_path(head, "QM9"))
        cfg.seed = repeat
        model = load_head_from_config(cfg)
        if head == "linear":
            model.fit(scaler.transform(x_all[fit]), y[fit])
        else:
            model.fit(scaler.transform(x_all[fit]), y[fit], x_val=scaler.transform(x_all[val]), y_val=y[val])
        pv = np.asarray(model.predict(scaler.transform(x_all[val]))).reshape(-1)
        pt = np.asarray(model.predict(scaler.transform(x_all[test]))).reshape(-1)
        row = {"support": support, "model": model_name, "head": head, "repeat": repeat,
               "validation_R2": float(r2_score(y[val].astype(float), pv.astype(float))),
               "test_R2": float(r2_score(y[test].astype(float), pt.astype(float)))}
        print(json.dumps(row), flush=True)
        if sink:
            sink.write(json.dumps(row) + "\n")
    if sink:
        sink.close()


@cli.command()
@click.option("--support", type=click.Choice(["100", "10000"]), required=True)
@click.option("--model-config", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--lora-rank", type=int, default=None)
@click.option("--repeat", type=int, required=True)
@click.option("--batch-size", type=int, default=None,
              help="Default: 256 at support 100, 16 at support 10,000.")
@click.option("--output-dir", default=str(ROOT / "results/qm9_curve_adaptations"), show_default=True)
def finetune(support, model_config, lora_rank, repeat, batch_size, output_dir):
    """Full fine-tuning or LoRA with finetune_one.py on the cohort."""
    from finetune_small_sample import legacy_molformer_compat

    support = int(support)
    rows, _ = load_rows(support)
    module = types.ModuleType("utils.datasets")
    module.load_dataset = lambda name, *_, **__: [
        types.SimpleNamespace(smiles=r["smiles"], y=np.array([r["epsilon_lumo_eV"]], dtype=np.float32))
        for r in rows
    ]
    sys.modules["utils.datasets"] = module
    legacy_molformer_compat()
    import finetune_one

    finetune_one.TARGET_INDEX = {**finetune_one.TARGET_INDEX, "QM9.eps_lumo": 0}  # one label column
    regime = "full-data" if support == 10000 else "low-data"
    batch_size = batch_size or (16 if support == 10000 else 256)
    args = ["--model-config", model_config, "--target", "QM9.eps_lumo", "--seed", str(repeat),
            "--regime", regime, "--val-frac", "0.1", "--lr-schedule", "none",
            "--batch-size", str(batch_size), "--output-dir", output_dir]
    if lora_rank:
        args += ["--use-lora", "--lora-rank", str(lora_rank)]
    finetune_one.main.main(args=args, standalone_mode=False)


if __name__ == "__main__":
    cli()
