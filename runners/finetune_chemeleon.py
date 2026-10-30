"""Fine-tune CheMeleon end-to-end on the evaluation splits.

The pretrained D-MPNN is combined with mean aggregation and a fresh 256-unit
regression or binary classification head. Regression targets are standardized
on training data and predictions are unscaled. Training uses deterministic
Lightning, up to 30 epochs, with checkpoint selection on validation loss and no
hyperparameter optimization.

The random 80/20 split uses seed 2032025 + run; 10% of training is reserved for
validation. For QM9, the first 1,250 rows and all 16 targets are used together
for comparability with the frozen representation, although the original model
uses single-task training. Scores use raw targets: R2 for regression and ROC-AUC
for BACE. The configured PyTorch build lacks kernels for V100 GPUs, so runs use CPU.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

RANDOM_SEED = 2032025  # run_one_eval.py's constant; the split must match cell for cell
QM9_MAX_ELEMENTS = 1250
EPOCHS = 30
FFN_HIDDEN = 256


def _build_split(n: int, seed: int):
    from sklearn.model_selection import train_test_split

    from validation.splits import get_split_indices

    train_idx, test_idx = get_split_indices(n, strategy="random", test_size=0.2, seed=seed)
    # The validator carves val from the ARRAYS with this same call and random_state; carving the
    # index array instead gives the identical partition because the permutation depends only on
    # length and seed.
    tr_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=seed)
    return tr_idx, val_idx, test_idx


@click.command()
@click.option("--dataset", required=True)
@click.option("--seed", default=0, type=int)
@click.option("--out-dir", default="results_chemeleon_finetune", type=str)
@click.option("--epochs", default=EPOCHS, type=int)
@click.option("--dropout", default=0.0, type=float)
@click.option("--protocol", type=click.Choice(["grid", "uniform"]), default="grid",
              help="uniform = the uniform protocol: QM9 on the seeded 1,250 sample instead of the first 1,250.")
@click.option("--target", default=None,
              help="QM9 only: fine-tune on this ONE target (e.g. epsilon_lumo, U_0), as every other "
                   "fine-tune arm does, instead of all 16 jointly. Names as in TARGETS_NAMES order.")
def main(dataset: str, seed: int, out_dir: str, epochs: int, dropout: float, protocol: str,
         target: str | None) -> None:
    out_path = Path(out_dir) / dataset / (target or "") / f"seed{seed}.json"
    if out_path.is_file():
        click.echo(f"skip: {out_path} exists")
        return

    import numpy as np
    import torch
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from chemprop.models import MPNN
    from chemprop.nn import BinaryClassificationFFN, BondMessagePassing, RegressionFFN, UnscaleTransform
    from chemprop.nn.agg import MeanAggregation
    from lightning import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from sklearn.metrics import r2_score, roc_auc_score

    from utils.datasets import load_dataset, maybe_load_targets

    run_seed = RANDOM_SEED + seed
    seed_everything(run_seed, workers=True)

    ds = load_dataset(dataset)
    smiles = [d.smiles for d in ds]
    targets = maybe_load_targets(ds)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    elif targets.ndim == 3:
        targets = targets.reshape(targets.shape[0], -1)
    if dataset == "QM9" and protocol == "uniform":
        import uniform_protocol as UP
        if len(smiles) != UP.QM9_N_TOTAL:
            raise SystemExit("QM9 has %d rows, the uniform sample is drawn from %d" % (len(smiles), UP.QM9_N_TOTAL))
        _idx = UP.qm9_sample()
        smiles, targets = [smiles[i] for i in _idx], targets[_idx]
    elif dataset == "QM9":
        smiles, targets = smiles[:QM9_MAX_ELEMENTS], targets[:QM9_MAX_ELEMENTS]
    if target is not None:
        # The order R2_per_target has always been written in: the repo's own TARGETS_NAMES["QM9"].
        _order = ["mu", "alpha", "epsilon_homo", "epsilon_lumo", "gap_homo_lumo", "R2_avg", "ZPVE",
                  "U_0", "U_298", "H_298", "G_298", "C_v_298", "U_0_atom", "U_298_atom",
                  "H_298_atom", "G_298_atom"]
        if dataset != "QM9" or target not in _order:
            raise SystemExit("--target is a QM9 target name, one of %s" % _order)
        targets = np.asarray(targets)[:, [_order.index(target)]]
    targets = np.asarray(targets, dtype=np.float64)
    is_classification = dataset == "BACE"

    tr_idx, val_idx, te_idx = _build_split(len(smiles), run_seed)
    click.echo(f"{dataset} seed {seed}: {len(tr_idx)} train / {len(val_idx)} val / {len(te_idx)} test"
               f" of {len(smiles)}; targets {targets.shape[1]}")

    featurizer = SimpleMoleculeMolGraphFeaturizer()
    def make(idx):
        return MoleculeDataset([MoleculeDatapoint.from_smi(smiles[i], targets[i]) for i in idx], featurizer)

    train_ds, val_ds, test_ds = make(tr_idx), make(val_idx), make(te_idx)
    scaler = None
    if not is_classification:
        scaler = train_ds.normalize_targets()
        val_ds.normalize_targets(scaler)
    train_dl = build_dataloader(train_ds, num_workers=0, seed=run_seed)
    val_dl = build_dataloader(val_ds, num_workers=0, shuffle=False)
    test_dl = build_dataloader(test_ds, num_workers=0, shuffle=False)

    ckpt = torch.load(Path.home() / ".chemprop" / "chemeleon_mp.pt", weights_only=True)
    ckpt["hyper_parameters"]["dropout"] = dropout
    mp = BondMessagePassing(**ckpt["hyper_parameters"])
    mp.load_state_dict(ckpt["state_dict"])
    output_transform = (UnscaleTransform.from_standard_scaler(scaler) if scaler is not None
                        else torch.nn.Identity())
    ffn_cls = BinaryClassificationFFN if is_classification else RegressionFFN
    ffn = ffn_cls(n_tasks=targets.shape[1], input_dim=mp.output_dim, hidden_dim=FFN_HIDDEN,
                  dropout=dropout, output_transform=output_transform)
    model = MPNN(mp, MeanAggregation(), ffn)

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="cheft_") as tmp:
        checkpoint = ModelCheckpoint(dirpath=tmp, filename="best", monitor="val_loss", mode="min",
                                     save_last=False, auto_insert_metric_name=False)
        trainer = Trainer(max_epochs=epochs, accelerator="cpu", devices=1, deterministic=True,
                          logger=False, enable_progress_bar=False, enable_model_summary=False,
                          callbacks=[checkpoint], check_val_every_n_epoch=1)
        trainer.fit(model, train_dl, val_dl)
        best = checkpoint.best_model_path
        if not best or not os.path.exists(best):
            raise SystemExit("no checkpoint written; refusing to score the last epoch silently")
        # Selection on validation loss, never on test -- avoiding test-set selection.
        model = MPNN.load_from_checkpoint(best)
        preds = torch.cat(trainer.predict(model, test_dl)).numpy().reshape(len(te_idx), -1)

    y_true = targets[te_idx]
    metrics: dict[str, object] = {}
    if is_classification:
        metrics["ROC_AUC"] = float(roc_auc_score(y_true.ravel(), preds.ravel()))
    else:
        per_target = [float(r2_score(y_true[:, t], preds[:, t])) for t in range(y_true.shape[1])]
        metrics["R2_per_target"] = per_target
        metrics["R2"] = per_target[0] if len(per_target) == 1 else float(np.mean(per_target))
    record = {
        "dataset": dataset, "seed": seed, "run_seed": run_seed, "mode": "finetune_end_to_end",
        "protocol": protocol, "target": target,
        "epochs": epochs, "dropout": dropout, "ffn_hidden": FFN_HIDDEN,
        "n_train": len(tr_idx), "n_val": len(val_idx), "n_test": len(te_idx),
        "best_checkpoint_epoch": int(getattr(checkpoint, "best_model_score", float("nan")) is not None
                                     and trainer.current_epoch),
        "val_loss_best": float(checkpoint.best_model_score) if checkpoint.best_model_score is not None else None,
        "seconds": round(time.time() - started, 1),
        **metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    json.dump(record, open(tmp_path, "w"), indent=1)
    os.replace(tmp_path, out_path)
    click.echo(json.dumps({k: v for k, v in record.items() if k != "R2_per_target"}))


if __name__ == "__main__":
    main()
