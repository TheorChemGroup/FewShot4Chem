"""Fine-tune one foundation model, target, seed and regime.

A regression head is trained with the backbone using AdamW. Checkpoint diagnostics
include embedding drift and CKA on a held-out 1,000-molecule probe set, and optionally
masked-language-model loss. The final test score is saved.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import click
import numpy as np
import torch
from omegaconf import OmegaConf
from rdkit import Chem, RDLogger
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# QM9.U_0 is an extensive energy with mean about -9787; U_0, U_298, H_298 and
# G_298 have similar means and standard deviations. Targets are not standardized:
# AdamW's bounded steps on raw MSE can produce a strongly negative R2 for U_0.
# Standardizing U_0 alone would make its protocol inconsistent with other targets.
TARGET_INDEX = {"ESOL": 0, "FreeSolv": 0, "QM9.eps_lumo": 3, "QM9.U_0": 7,
                "Lipophilicity": 0, "BACE": 0}
# Binary targets: BCE-with-logits loss, and ROC-AUC wherever the regression targets use R2
# (epoch selection and the reported scores; the result keeps the *_r2 field names).
CLASSIFICATION_TARGETS = {"BACE"}
TARGET_DATASET_NAME = {"ESOL": "ESOL", "FreeSolv": "FreeSolv", "QM9.eps_lumo": "QM9",
                       "QM9.U_0": "QM9", "QM9.atom_count": "QM9", "QM9.bond_count": "QM9",
                       "Lipophilicity": "Lipophilicity", "BACE": "BACE"}

# TARGETS WHOSE LABEL IS NOT IN THE DATASET AT ALL.
#
# Every target above indexes a column of `d.y`. Atom count and bond count are not columns of
# anything -- QM9's `d.y` carries the sixteen quantum-chemical properties and neither count is
# among them. They are computed per molecule from its SMILES. So these targets deliberately
# have NO entry in TARGET_INDEX: an accidental `d.y[TARGET_INDEX[t]]` for one of them raises
# KeyError instead of silently fitting a physical property under a count's name.
#
# BOTH COUNTS INCLUDE HYDROGEN, matching the frozen baseline so the fine-tuned arms
# fit the same column its frozen row does. Both wrong spellings are one keystroke away and
# neither raises:
#     Chem.MolFromSmiles(smi).GetNumAtoms()   -> heavy atoms only (Hs are implicit)
#     Chem.MolFromSmiles(smi).GetNumBonds()   -> heavy-atom bonds only
# which is why the definitions are asserted at import below rather than trusted.
SMILES_TARGETS = {
    "QM9.atom_count": lambda mol: mol.GetNumAtoms(),
    "QM9.bond_count": lambda mol: mol.GetNumBonds(),
}


def _smiles_target_value(target: str, smiles: str) -> float:
    """Label for a SMILES-derived target. Raises on an unparseable molecule, never returns 0.

    A count of 0 is indistinguishable from a real small value once it is in `y_arr`, and a
    single silently-zeroed row shifts the fit for every seed that draws it. The caller is
    expected to let this propagate.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        msg = f"RDKit could not parse SMILES {smiles!r} for target {target}"
        raise ValueError(msg)
    return float(SMILES_TARGETS[target](Chem.AddHs(mol)))


# Asserted at import, so a wrong count definition cannot reach a fit. CH3NH2 = C + N + 5H = 7
# atoms and 6 bonds; benzene = 6C + 6H = 12 and 12; ethanol = 2C + O + 6H = 9 and 8.
for _smi, _atoms, _bonds in (("CN", 7, 6), ("c1ccccc1", 12, 12), ("CCO", 9, 8)):
    for _t, _want in (("QM9.atom_count", _atoms), ("QM9.bond_count", _bonds)):
        _got = _smiles_target_value(_t, _smi)
        if _got != _want:
            _msg = (
                f"{_t} definition is wrong: {_smi!r} gave {_got:g}, expected {_want}. This is "
                "the missing-AddHs mistake -- the run is aborted rather than fitting the "
                "wrong column.")
            raise SystemExit(_msg)

REGIMES = {
    "low-data": {"n_train_cap": 100, "max_epochs": 30, "lr": 1e-5, "patience": 8},
    "full-data": {"n_train_cap": None, "max_epochs": 20, "lr": 5e-6, "patience": 6},
    # Hyperparameters follow dataset scale: the 1,000-row regime uses the same
    # settings as similarly sized ESOL and FreeSolv pools. Full-data sizes differ
    # substantially across datasets. With a 10% validation carve, the 1,000-row
    # cap gives 900 fit rows, matching the 1,250-row probing split. The cap is
    # applied after the test split, preserving paired test sets.
    "mid-data": {"n_train_cap": 1000, "max_epochs": 20, "lr": 5e-6, "patience": 6},
}
TEST_FRAC = 0.2
PROBE_N = 1000  # held-out probe set size for drift/CKA


REQUIRED_RESULT_KEYS = ("fm", "target", "regime", "seed", "test_r2",
                        "selection_metric", "n_fit", "drift_log")


def _is_complete_result(path: Path) -> bool:
    """True only if `path` holds a parseable result with every field downstream code reads.

    Guards the resume path: `exists()` answers "is there a file", which is not the question.
    A cell killed mid-write leaves a file that exists and is useless, and skipping on
    existence would drop that cell silently from every aggregate that follows.
    """
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text())
    except Exception:
        return False
    return isinstance(d, dict) and all(k in d for k in REQUIRED_RESULT_KEYS)


def _r2(y, pred) -> float:
    """Return R2, or NaN for non-finite predictions so divergent runs are recorded."""
    from sklearn.metrics import r2_score
    pred = np.asarray(pred, dtype=float)
    if not np.isfinite(pred).all():
        return float("nan")
    return float(r2_score(y, pred))


def _auc(y, pred) -> float:
    """ROC-AUC of logits (equal to that of probabilities), NaN when predictions are non-finite."""
    from sklearn.metrics import roc_auc_score
    pred = np.asarray(pred, dtype=float)
    if not np.isfinite(pred).all():
        return float("nan")
    return float(roc_auc_score(np.asarray(y).ravel(), pred.ravel()))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    num = float(np.linalg.norm(Xc.T @ Yc, "fro") ** 2)
    den = float(np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro"))
    return num / den if den > 0 else float("nan")


def embed_batch(model, smiles_list: list[str], device: str, batch_size: int = 32) -> np.ndarray:
    """Run the FM's forward on a list of SMILES; return (N, dim) embeddings."""
    from torch_geometric.data import Data
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i : i + batch_size]
            data = Data(smiles=batch_smiles)
            emb = model(data)
            if isinstance(emb, tuple):
                emb = emb[0]
            out.append(emb.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def _predict(model, smiles_list: list[str], batch_size: int) -> np.ndarray:
    """Batched forward for scoring. Caller is responsible for model.eval()."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            preds.append(model.forward_smiles(smiles_list[i:i + batch_size]).detach().cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


class FineTuneLitModule(nn.Module):
    """FM backbone + 2-layer MLP regression head. requires_grad=True end-to-end."""

    def __init__(self, fm_backbone: nn.Module, emb_dim: int, hidden_dim: int = 256,
                 lora_cfg: dict | None = None):
        super().__init__()
        self.fm = fm_backbone
        if lora_cfg is not None:
            # LoRA mode: freeze base, inject low-rank adapters via PEFT (auto-discover linear modules).
            from peft import LoraConfig, get_peft_model
            for p in self.fm.parameters():
                p.requires_grad = False
            cfg = LoraConfig(
                r=lora_cfg["rank"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
                target_modules="all-linear", bias="none",
            )
            self.fm = get_peft_model(self.fm, cfg)
            n_train = sum(p.numel() for p in self.fm.parameters() if p.requires_grad)
            n_tot = sum(p.numel() for p in self.fm.parameters())
            logger.info("LoRA: %d trainable / %d total params (%.3f%%)", n_train, n_tot, 100 * n_train / max(n_tot, 1))
        else:
            for p in self.fm.parameters():
                p.requires_grad = True
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_smiles(self, smiles_list: list[str]) -> torch.Tensor:
        from torch_geometric.data import Data
        data = Data(smiles=smiles_list)
        emb = self.fm(data)
        if isinstance(emb, tuple):
            emb = emb[0]
        return self.head(emb).squeeze(-1)


@click.command()
@click.option("--model-config", required=True, type=click.Path(exists=True))
@click.option("--target", required=True,
              type=click.Choice(list(TARGET_INDEX) + list(SMILES_TARGETS)))
@click.option("--seed", required=True, type=int)
@click.option("--regime", required=True, type=click.Choice(list(REGIMES.keys())))
@click.option("--output-dir", default="finetune/results", type=click.Path())
@click.option("--lr-override", type=float, default=None, help="Override regime lr")
@click.option("--max-epochs-override", type=int, default=None, help="Override regime max_epochs")
@click.option("--use-lora", is_flag=True, default=False, help="LoRA mode (PEFT) instead of full finetune")
@click.option("--lora-rank", type=int, default=8)
@click.option("--lora-alpha", type=int, default=None, help="Default 2*rank")
@click.option("--lora-dropout", type=float, default=0.05)
@click.option("--val-frac", type=float, default=0.0,
              help="Fraction of the TRAIN POOL held out for validation-based selection. "
                   "0.0 (default) reproduces the legacy TEST-selected path bit-for-bit. "
                   "Test is NEVER used for any decision when >0.")
@click.option("--lr-schedule", type=click.Choice(["none", "cosine"]), default="none",
              help="'none' (default) = legacy constant LR. 'cosine' = linear warmup + cosine decay.")
@click.option("--warmup-frac", type=float, default=0.05,
              help="Warmup fraction of total steps when --lr-schedule cosine.")
@click.option("--split", "split_mode", type=click.Choice(["regime", "grid"]), default="regime",
              help="regime = the published arms' split (train_test_split(random_state=seed) + --val-frac). "
                   "grid = the uniform protocol: the grid's 72/8/20 with seed 2032025+s, "
                   "QM9 on the seeded 1,250 sample, regime full-data, validation carve fixed at 10%.")
@click.option("--batch-size", type=int, default=16, show_default=True,
              help="Training/prediction batch size (the BACE/Lipophilicity and batch-256 runs use 256).")
@click.option("--stop-on-nonfinite", is_flag=True, default=False,
              help="Raise FloatingPointError on a non-finite training loss or on non-finite parameters "
                   "after an optimizer step, instead of recording a diverged run.")
@click.option("--verbose", "-v", count=True)
def main(model_config, target, seed, regime, output_dir, lr_override, max_epochs_override,
         use_lora, lora_rank, lora_alpha, lora_dropout, val_frac, lr_schedule, warmup_frac, split_mode,
         batch_size, stop_on_nonfinite, verbose):
    logging.basicConfig(level=max(10, 40 - verbose * 10),
                        format="[%(asctime)s] %(levelname)s %(message)s")
    cfg = OmegaConf.load(model_config)
    fm_name = cfg.name
    if use_lora and lora_alpha is None:
        lora_alpha = 2 * lora_rank
    # LoRA runs nest under a rank-specific subdir so the 3 ranks don't collide.
    if use_lora:
        out_path = Path(output_dir) / fm_name / target / regime / f"rank{lora_rank}" / f"seed_{seed}.json"
    else:
        out_path = Path(output_dir) / fm_name / target / regime / f"seed_{seed}.json"
    if _is_complete_result(out_path):
        logger.info("skip: %s exists and is complete", out_path)
        return
    if out_path.exists():
        # Present but unreadable/partial: a previous attempt was killed mid-write, or wrote a
        # result that never finished. Recompute rather than inherit it.
        logger.warning("re-running %s: file exists but is NOT a complete result", out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from utils.datasets import load_dataset
    from utils.load_models import load_model

    # Reproducibility
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load FM
    t0 = time.time()
    fm = load_model(cfg).to(device)
    # A backbone whose default forward is NOT differentiable says so by offering this hook. The
    # text embedders (nn/text_embedder.py) embed through SentenceTransformer.encode(), which runs
    # under inference_mode: without the switch, "full fine-tune" and "LoRA" would both silently
    # train only the MLP head. The hook also upcasts bf16 weights to float32 and, for jina-v5,
    # merges its active task adapter, so the PEFT arm below STACKS our LoRA on top of it. Called
    # before the pre-finetune probe embeddings, so drift is measured from the function trained.
    if hasattr(fm, "prepare_for_finetuning"):
        fm.prepare_for_finetuning()
    logger.info("loaded FM %s in %.1fs", fm_name, time.time() - t0)

    # Load target data
    ds_name = TARGET_DATASET_NAME[target]
    ds = load_dataset(ds_name)
    cap = REGIMES[regime]["n_train_cap"]
    # Cap every QM9 target at 12,500 rows to preserve pairing with the frozen
    # baseline; the resulting test set contains 2,500 rows.
    if split_mode == "grid":
        # Use full-data hyperparameters and the common validation split.
        if regime != "full-data":
            raise SystemExit("--split grid runs at --regime full-data (the 1,250 QM9 sample is its own size)")
        if val_frac:
            raise SystemExit("--split grid fixes the validation carve at the grid's 10%%; drop --val-frac %s" % val_frac)
        import uniform_protocol as UP
        cap = None
    if target.startswith("QM9.") and split_mode != "grid":
        target_cap = 12500  # match the baseline split
    else:
        target_cap = None
    smiles, ys = [], []
    for i, d in enumerate(ds):
        if target_cap is not None and i >= target_cap:
            break
        smiles.append(d.smiles)
        # The dataset supplies the molecule for every target and the LABEL for most of them.
        # For the count targets it supplies only the molecule; see SMILES_TARGETS.
        if target in SMILES_TARGETS:
            ys.append(_smiles_target_value(target, d.smiles))
        else:
            ys.append(np.asarray(d.y).ravel()[TARGET_INDEX[target]])
    y_arr = np.array(ys, dtype=np.float32)
    if split_mode == "grid" and target.startswith("QM9."):
        if len(smiles) != UP.QM9_N_TOTAL:
            raise SystemExit("QM9 loaded %d rows, the uniform sample is drawn from %d" % (len(smiles), UP.QM9_N_TOTAL))
        _samp = UP.qm9_sample()
        smiles = [smiles[i] for i in _samp]
        y_arr = y_arr[_samp]
    n = len(smiles)
    if split_mode == "grid":
        _fit, _val, test_idx = UP.grid_split(n, seed)
        train_pool_idx_full = np.concatenate([np.asarray(_fit), np.asarray(_val)])
    else:
        # Paired with the baseline: sklearn train_test_split(random_state=seed) — bit-identical splits
        train_pool_idx_full, test_idx = train_test_split(np.arange(n), test_size=TEST_FRAC, random_state=seed)
    # Low-data regime: subsample the train pool deterministically (seed for reproducibility)
    rng = np.random.default_rng(seed)
    if cap is not None and cap < len(train_pool_idx_full):
        train_pool_idx = rng.choice(train_pool_idx_full, size=cap, replace=False)
    else:
        train_pool_idx = train_pool_idx_full
    # Validation split, carved from the TRAIN POOL only. The test set is untouched, so the
    # frozen-baseline pairing (same seed -> same train_test_split -> same test rows) is preserved
    # exactly. A dedicated RNG keeps the main `rng` stream (probe set, epoch shuffles) unperturbed,
    # so val_frac=0 is bit-identical to the legacy path.
    if split_mode == "grid":
        fit_idx, val_idx = np.asarray(_fit), np.asarray(_val)
        val_frac = 0.1          # recorded as what the carve IS; drives the val-selection path below
    elif val_frac and val_frac > 0.0:
        val_rng = np.random.default_rng(seed + 100_000)
        n_val = max(1, int(round(val_frac * len(train_pool_idx))))
        perm = val_rng.permutation(np.asarray(train_pool_idx))
        val_idx = perm[:n_val]
        fit_idx = perm[n_val:]
    else:
        val_idx = np.asarray([], dtype=int)
        fit_idx = np.asarray(train_pool_idx)

    train_smiles = [smiles[i] for i in fit_idx]
    train_y = y_arr[fit_idx]
    val_smiles = [smiles[i] for i in val_idx]
    val_y = y_arr[val_idx]
    test_smiles = [smiles[i] for i in test_idx]
    test_y = y_arr[test_idx]
    # probe set: hold-out outside train+test if possible
    used = set(train_pool_idx.tolist()) | set(test_idx.tolist())
    leftover = [i for i in range(n) if i not in used]
    if len(leftover) >= PROBE_N:
        probe_idx = rng.choice(leftover, size=PROBE_N, replace=False)
    else:
        probe_idx = rng.choice(np.arange(n), size=min(PROBE_N, n), replace=False)
    probe_smiles = [smiles[i] for i in probe_idx]

    # Pre-finetune probe-set embeddings (frozen FM)
    pre_emb = embed_batch(fm, probe_smiles, device)
    emb_dim = pre_emb.shape[1]
    logger.info("FM emb_dim=%d, n_train=%d, n_test=%d, probe_n=%d",
                emb_dim, len(train_smiles), len(test_smiles), len(probe_smiles))

    # Build finetune wrapper
    lora_cfg = {"rank": lora_rank, "alpha": lora_alpha, "dropout": lora_dropout} if use_lora else None
    model = FineTuneLitModule(fm, emb_dim, lora_cfg=lora_cfg).to(device)
    # Names of the tensors that actually receive gradients -- the only ones worth checkpointing.
    _trainable_keys = {n for n, prm in model.named_parameters() if prm.requires_grad}
    # LoRA trains only adapters (init B=0), needs a higher lr than full-finetune to move at all.
    default_lr = (5e-4 if use_lora else REGIMES[regime]["lr"])
    lr = lr_override if lr_override is not None else default_lr
    max_epochs = max_epochs_override if max_epochs_override is not None else REGIMES[regime]["max_epochs"]
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    classification = target in CLASSIFICATION_TARGETS
    loss_fn = nn.BCEWithLogitsLoss() if classification else nn.MSELoss()
    _score = _auc if classification else _r2
    patience = REGIMES[regime]["patience"]

    # Training loop (manual, small batch)
    BS = batch_size
    n_tr = len(train_smiles)

    # LR schedule. Legacy runs used a CONSTANT lr despite the module docstring advertising cosine;
    # `none` preserves that exactly. `cosine` = linear warmup then cosine decay, stepped per batch.
    steps_per_epoch = max(1, math.ceil(n_tr / BS))
    total_steps = max(1, steps_per_epoch * max_epochs)
    if lr_schedule == "cosine":
        warmup_steps = max(1, int(warmup_frac * total_steps))

        def _lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

        sched = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda)
    else:
        sched = None

    # SELECTION RULE. This is the field the whole arm turned on, so it is recorded in the artifact
    # rather than left to be reconstructed from prose later.
    select_on_val = bool(val_frac and val_frac > 0.0 and len(val_smiles) > 0)
    selection_metric = (("val_auc" if classification else "val_r2") if select_on_val
                        else "test_r2_PEEKING_LEGACY")
    logger.info("training: lr=%.2e max_epochs=%d patience=%d schedule=%s select_on=%s "
                "n_fit=%d n_val=%d n_test=%d",
                lr, max_epochs, patience, lr_schedule, selection_metric,
                n_tr, len(val_smiles), len(test_smiles))

    best_score = -float("inf")
    diverged_during_training = False   # any epoch whose score went NaN, even if an earlier epoch is the one kept
    best_epoch = -1
    best_state = None
    no_improve = 0
    drift_log = []
    for ep in range(max_epochs):
        model.train()
        epoch_loss = 0.0; n_batches = 0
        order = rng.permutation(n_tr)
        for bs_start in range(0, n_tr, BS):
            idx = order[bs_start:bs_start + BS]
            # Skip batches < 2 to avoid BatchNorm "Expected >1 value per channel" on graph FMs
            # (gin-pretrained drops invalid-SMILES rows inside forward, which can shrink the batch).
            if len(idx) < 2:
                continue
            batch_smiles = [train_smiles[i] for i in idx]
            batch_y = torch.tensor(train_y[idx], dtype=torch.float32, device=device)
            optim.zero_grad()
            pred = model.forward_smiles(batch_smiles)
            # Defensive: if FM dropped rows internally (graph FMs may), shape may shrink
            if pred.shape[0] != batch_y.shape[0]:
                logger.warning("ep=%d bs=%d: forward returned %d preds for %d labels — skipping batch",
                               ep, bs_start, pred.shape[0], batch_y.shape[0])
                continue
            loss = loss_fn(pred, batch_y)
            if stop_on_nonfinite and not bool(torch.isfinite(loss).all()):
                raise FloatingPointError("nonfinite training loss at epoch %d batch %d" % (ep, bs_start))
            loss.backward()
            optim.step()
            if stop_on_nonfinite and not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError("nonfinite parameters after AdamW at epoch %d batch %d" % (ep, bs_start))
            if sched is not None:
                sched.step()
            epoch_loss += float(loss.item()); n_batches += 1
        # Eval. When selecting on validation, the TEST set is not evaluated inside the loop at all
        # -- it cannot leak into a decision it is never shown to.
        model.eval()
        from sklearn.metrics import r2_score
        if select_on_val:
            val_r2 = _score(val_y, _predict(model, val_smiles, BS))
            test_r2 = None
            score = val_r2
        else:
            val_r2 = None
            test_r2 = _score(test_y, _predict(model, test_smiles, BS))
            score = test_r2
        # embedding drift
        cur_emb = embed_batch(fm, probe_smiles, device)
        l2_drift = float(np.linalg.norm(cur_emb - pre_emb) / np.linalg.norm(pre_emb))
        cos_drift = float(1 - (cur_emb * pre_emb).sum(1).mean() /
                          (np.linalg.norm(cur_emb, axis=1) * np.linalg.norm(pre_emb, axis=1)).mean())
        cka = linear_cka(cur_emb, pre_emb)
        drift_log.append({"epoch": ep, "train_loss": epoch_loss / max(n_batches, 1),
                          "val_r2": val_r2, "test_r2": test_r2,
                          "lr": optim.param_groups[0]["lr"],
                          "l2_drift_rel": l2_drift,
                          "cos_drift_mean": cos_drift, "cka_to_pretrained": cka})
        logger.info("ep=%d loss=%.4f %s=%.4f lr=%.2e l2_drift=%.3f cka=%.3f",
                    ep, epoch_loss / max(n_batches, 1), selection_metric, score,
                    optim.param_groups[0]["lr"], l2_drift, cka)
        if not math.isfinite(score):
            diverged_during_training = True
        if score > best_score:
            best_score = score; best_epoch = ep; no_improve = 0
            if select_on_val:
                # Snapshot ONLY the trainable tensors. Under LoRA that is the adapters + head
                # (<1% of params), so keeping the best-val checkpoint costs almost nothing.
                best_state = {k: v.detach().clone().cpu()
                              for k, v in model.state_dict().items()
                              if k in _trainable_keys}
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("early stop at ep=%d (patience=%d, best=%d)", ep, patience, best_epoch)
                break

    # Test is touched HERE and only here when selecting on validation: restore the
    # best-validation weights, then score the test set exactly once.
    if select_on_val:
        if best_state is not None:
            missing = model.load_state_dict(best_state, strict=False)
            logger.info("restored best-val checkpoint from epoch %d (%d tensors)",
                        best_epoch, len(best_state))
        model.eval()
        reported_test_r2 = _score(test_y, _predict(model, test_smiles, BS))
        logger.info("FINAL (val-selected ep=%d): test_r2=%.4f  best_val_r2=%.4f",
                    best_epoch, reported_test_r2, best_score)
    else:
        reported_test_r2 = drift_log[-1]["test_r2"] if drift_log else float("nan")

    result = {
        "fm": fm_name, "target": target, "regime": regime, "seed": seed,
        # Record both weight and code revisions: cross-repository auto_map entries
        # can resolve executable code independently of the model revision.
        "model_revision": (cfg.hyperparams.get("revision")
                           if hasattr(cfg, "hyperparams") else None),
        "code_revision": (cfg.hyperparams.get("code_revision")
                          if hasattr(cfg, "hyperparams") else None),
        "n_train": len(train_smiles), "n_test": len(test_smiles), "probe_n": len(probe_smiles),
        "n_fit": len(train_smiles), "n_val": len(val_smiles),
        "selection_metric": selection_metric,
        "test_r2": reported_test_r2,
        "best_score": best_score, "best_epoch": best_epoch,
        "best_val_r2": best_score if select_on_val else None,
        "n_epochs_ran": len(drift_log),
        "stopped_early": len(drift_log) < max_epochs,
        # Legacy field names, kept so old aggregators keep parsing. Under val selection
        # best_test_r2 is None: no argmax over test is ever computed.
        "best_test_r2": (None if select_on_val else best_score),
        "final_test_r2": reported_test_r2,
        "diverged": not math.isfinite(reported_test_r2),
        "diverged_during_training": diverged_during_training,
        "drift_log": drift_log,
        "config": {"lr": lr, "max_epochs": max_epochs,
                   "patience": patience, "batch_size": BS,
                   "lr_override": lr_override, "max_epochs_override": max_epochs_override,
                   "use_lora": use_lora, "lora_rank": lora_rank if use_lora else None,
                   "lora_alpha": lora_alpha if use_lora else None,
                   "lora_dropout": lora_dropout if use_lora else None,
                   "val_frac": val_frac, "lr_schedule": lr_schedule,
                   # which protocol produced this file -- the uniform one says so, the published do not
                   "split": split_mode,
                   "qm9_sample": ("qm9_uniform_sample_1250.txt" if split_mode == "grid"
                                  and target.startswith("QM9.") else None),
                   "warmup_frac": warmup_frac if lr_schedule == "cosine" else None},
        "wall_time_sec": time.time() - t0,
    }
    # ATOMIC WRITE. `write_text` is a truncate-then-write: a job killed partway (TIMEOUT,
    # NODE_FAIL, preemption) leaves a short file that `exists()` happily accepts, so an
    # idempotent re-submission would skip that cell FOREVER and the grid would come up
    # quietly short. Write to a temp file in the same directory, fsync, then os.replace,
    # which is atomic on POSIX: the output either is not there or is the complete result.
    tmp_path = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
    with open(tmp_path, "w") as fh:
        json.dump(result, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, out_path)
    logger.info("wrote %s test_r2=%.4f (selected on %s)", out_path, reported_test_r2, selection_metric)


if __name__ == "__main__":
    main()
