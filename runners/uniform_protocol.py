"""Shared evaluation protocol.

Use a 72/8/20 train/validation/test split, with test_size=0.2 and seed
2032025+s, followed by a 10% validation carve using the same seed. Draw
1,250 QM9 molecules without replacement from all 129,425 using seed 2032025,
and retain dataset order. This avoids the size bias of the ordered prefix;
the saved indices must agree with the generated sample. Select heads using
validation scores, use the same head pool for each representation, and compute
paired intervals over five seeds.
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runners"))

RANDOM_SEED = 2032025
N_SEEDS = 5
QM9_N_TOTAL = 129_425
QM9_SAMPLE_N = 1250
QM9_SAMPLE_FILE = Path(__file__).resolve().parent / "qm9_uniform_sample_1250.txt"
REGRESSION_HEADS = ("linear", "xgboost", "mlp", "gp-matern", "gp-rbf", "tabicl", "tabpfn")
CLASSIFICATION_HEADS = ("logistic", "mlp", "xgboost", "tabpfn", "tabicl")
MORGAN = "morgan-2048-r2"   # ECFP4, fixed in advance; the only fingerprint drawn


def grid_split(n: int, s: int):
    """(fit, val, test) index arrays for seed index s in 0..4 -- the grid's own split."""
    from finetune_chemeleon import _build_split
    if not 0 <= s < N_SEEDS:
        raise ValueError("seed index %r outside 0..%d" % (s, N_SEEDS - 1))
    return _build_split(n, RANDOM_SEED + s)


def _draw_qm9_sample() -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    return np.sort(rng.choice(QM9_N_TOTAL, size=QM9_SAMPLE_N, replace=False))


def qm9_sample() -> np.ndarray:
    """The 1,250 QM9 row indices, ascending. Drawn from the rule AND read from the committed file;
    a disagreement means one of them moved, and every QM9 number would move with it."""
    drawn = _draw_qm9_sample()
    if QM9_SAMPLE_FILE.exists():
        stored = np.array([int(x) for x in QM9_SAMPLE_FILE.read_text().split()], dtype=np.int64)
        if not np.array_equal(stored, drawn):
            raise SystemExit("%s disagrees with the rng(%d) draw -- refusing" % (QM9_SAMPLE_FILE, RANDOM_SEED))
    return drawn


def select_head(val_by_head: dict[str, dict[int, float]]) -> str:
    """The head with the best MEAN VALIDATION score over the seeds -- chosen once per (model, row).

    Every head must carry all five seeds: a head short of seeds would be averaged over a different
    set of splits from its rivals, and the argmax would compare unlike things.
    """
    short = {h: len(v) for h, v in val_by_head.items() if len(v) != N_SEEDS}
    if short:
        raise ValueError("heads without all %d seeds: %s" % (N_SEEDS, short))
    return max(val_by_head, key=lambda h: st.mean(val_by_head[h].values()))


if __name__ == "__main__":
    idx = _draw_qm9_sample()
    QM9_SAMPLE_FILE.write_text("\n".join(str(int(i)) for i in idx) + "\n")
    print("wrote %s: %d indices, min %d, max %d, median %d"
          % (QM9_SAMPLE_FILE.name, len(idx), idx.min(), idx.max(), int(np.median(idx))))
