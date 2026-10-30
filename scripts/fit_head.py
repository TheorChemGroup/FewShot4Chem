"""Fit a prediction head on frozen features for the small-sample (Figure 1) protocol.

One cell = (dataset, representation, head, repeat). Each dataset has a fixed cohort of 100
molecules (`data/cohorts/small_sample.json`); repeat *s* splits it 50 fit / 10 validation /
40 test. Features are standardised on the 60 fit+validation molecules, the head is fitted on the
50 fit molecules (validation is passed to heads that early-stop), and the script reports
validation and test scores (R² for regression, ROC-AUC for BACE).

Features come from `scripts/featurize.py` (`features/small_sample/<dataset>__<model>.npy`, rows
in cohort order).

    python scripts/fit_head.py --dataset ESOL --model rdkit-descriptors --head linear --repeats 0-14
"""

import json
import sys
import time
from pathlib import Path

import click
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COHORTS = ROOT / "data/cohorts/small_sample.json"
FEATURES = ROOT / "features/small_sample"
HEADS = ("linear", "mlp", "xgboost", "gp-rbf", "gp-matern", "tabpfn", "tabicl")
# Properties scored per dataset in Figure 1.
PROPERTIES = {
    "BACE": ["active"],
    "Lipophilicity": ["logD"],
    "ESOL": ["logS"],
    "FreeSolv": ["dG_hydr"],
    "QM9": ["epsilon_lumo", "U_0"],
}
CLASSIFICATION = {"BACE"}
SEED_OFFSET = 2032025


def head_config_path(head, dataset):
    """Head hyperparameters used for every Figure 1 fit."""
    if dataset in CLASSIFICATION:
        return ROOT / f"models/configs/heads/{head}-classification.yaml"
    name = {"linear": "linear-regression", "xgboost": "xgboost-regression"}.get(head, head)
    return ROOT / f"models/configs/heads/{name}.yaml"


def parse_repeats(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def split_indices(cohort, repeat):
    split = cohort["splits"][str(repeat)]
    fit, val, test = (np.asarray(split[k], dtype=int) for k in ("fit", "val", "test"))
    assert (len(fit), len(val), len(test)) == (50, 10, 40)
    # The 60 non-test molecules: the same set the features are standardised on.
    pool = np.random.RandomState(SEED_OFFSET + repeat).permutation(100)[40:]
    assert set(pool) == set(fit) | set(val)
    return fit, val, test, pool


def make_head(head, dataset, repeat, tabpfn_weights=None, tabicl_weights=None):
    from omegaconf import OmegaConf

    from validation.heads import load_head_from_config

    cfg = OmegaConf.load(head_config_path(head, dataset))
    if "seed" in cfg:
        cfg.seed = SEED_OFFSET + repeat
    model = load_head_from_config(cfg)
    classification = dataset in CLASSIFICATION
    # The paper's TabPFN / TabICL fits use pinned local checkpoints (TabPFN 9.2.0 with the v3.5
    # weights, TabICL 2.2.0 with the v2 weights), not whatever the package would download.
    if cfg.head_type == "tabpfn" and tabpfn_weights:
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        factory = TabPFNClassifier if classification else TabPFNRegressor
        model._make_estimator = lambda: factory(  # noqa: SLF001
            model_path=str(tabpfn_weights),
            device=model.device,
            random_state=int(cfg.seed),
            ignore_pretraining_limits=True,
        )
    if cfg.head_type == "tabicl" and tabicl_weights:
        from tabicl import TabICLClassifier, TabICLRegressor

        factory = TabICLClassifier if classification else TabICLRegressor
        weights = Path(tabicl_weights)
        model._make_estimator = lambda: factory(  # noqa: SLF001
            model_path=str(weights),
            checkpoint_version=weights.name,
            allow_auto_download=False,
            device=model.device,
            random_state=int(cfg.seed),
        )
    return model


def fit_cell(dataset, model_name, head, prop, repeat, cohort, x, **weights):
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from validation.utils import calculate_sample_weights

    y = np.array([float(r[prop]) for r in cohort["records"]]).reshape(-1, 1)
    fit, val, test, pool = split_indices(cohort, repeat)
    x = StandardScaler().fit(x[pool]).transform(x)
    model = make_head(head, dataset, repeat, **weights)
    kwargs = {}
    if dataset in CLASSIFICATION:
        kwargs = {
            "sample_weight": calculate_sample_weights(y[fit]),
            "sample_weight_eval_set": (calculate_sample_weights(y[val]),),
        }
    start = time.time()
    model.fit(x[fit], y[fit], x_val=x[val], y_val=y[val], **kwargs)
    if dataset in CLASSIFICATION:
        pv = np.asarray(model.predict_proba(x[val]))[:, 1]
        pt = np.asarray(model.predict_proba(x[test]))[:, 1]
        vs, ts = roc_auc_score(y[val].ravel(), pv), roc_auc_score(y[test].ravel(), pt)
    else:
        pv = np.asarray(model.predict(x[val])).reshape(-1)
        pt = np.asarray(model.predict(x[test])).reshape(-1)
        vs, ts = r2_score(y[val, 0], pv), r2_score(y[test, 0], pt)
    return {
        "dataset": dataset,
        "property": prop,
        "model": model_name,
        "head": head,
        "repeat": repeat,
        "validation": float(vs),
        "test": float(ts),
        "metric": "ROC_AUC" if dataset in CLASSIFICATION else "R2",
        "validation_predictions": pv.tolist(),
        "test_predictions": pt.tolist(),
        "test_dataset_indices": [cohort["records"][i]["dataset_index"] for i in test],
        "seconds": time.time() - start,
    }


@click.command()
@click.option("--dataset", type=click.Choice(sorted(PROPERTIES)), required=True)
@click.option("--model", "model_name", required=True, help="Representation, e.g. rdkit-descriptors.")
@click.option("--head", type=click.Choice(HEADS), required=True)
@click.option("--property", "prop", default=None, help="Default: every Figure 1 property.")
@click.option("--repeats", default="0-14", show_default=True)
@click.option("--tabpfn-weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--tabicl-weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--out", type=click.Path(dir_okay=False), default=None, help="JSON-lines output.")
def main(dataset, model_name, head, prop, repeats, tabpfn_weights, tabicl_weights, out):
    cohort = json.loads(COHORTS.read_text())[dataset]
    x = np.load(FEATURES / f"{dataset}__{model_name}.npy")
    assert x.shape[0] == len(cohort["records"]) == 100, x.shape
    weights = {"tabpfn_weights": tabpfn_weights, "tabicl_weights": tabicl_weights}
    sink = open(out, "a") if out else None  # noqa: SIM115
    for p in [prop] if prop else PROPERTIES[dataset]:
        for repeat in parse_repeats(repeats):
            row = fit_cell(dataset, model_name, head, p, repeat, cohort, x, **weights)
            print(f"{dataset} {p} {model_name} {head} repeat={repeat} "
                  f"val={row['validation']:.6f} test={row['test']:.6f}", flush=True)
            if sink:
                sink.write(json.dumps(row) + "\n")
    if sink:
        sink.close()


if __name__ == "__main__":
    main()
