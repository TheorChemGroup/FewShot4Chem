"""Regenerate every cohort and split from its sampling rule and check the shipped files.

    python scripts/make_cohorts.py

All draws use seed 2032025 (and per-repeat offsets) and depend only on dataset sizes, so this
needs no data download. Cohorts:

* small-sample (Figure 1 and the Figure 2 support-60 points): per dataset, 100 molecules,
  `default_rng(2032025).choice(len(table), 100)` sorted; QM9's table is the uniform 1,250 sample.
  Repeat s: test = first 40 of `RandomState(2032025+s).permutation(100)`; of the other 60,
  `RandomState(2032025+s).permutation(60)` picks 10 validation, 50 fit.
  -> data/cohorts/small_sample.json
* QM9 uniform 1,250 (Figure 2 support 1,000): `default_rng(2032025).choice(129425, 1250)` sorted.
  -> runners/qm9_uniform_sample_1250.txt
* QM9 random 12,500 (support 10,000): `default_rng(2032025).choice(129425, 12500)` sorted;
  repeat s: `train_test_split(test_size=0.2, random_state=s)`, then the first 1,000 of
  `default_rng(s+100000).permutation(pool)` are validation (finetune_one.py's own split).
  -> data/cohorts/qm9_random12500_indices.txt
* QM9 random 125 (support 100), nested in the 12,500: `default_rng(2032025).choice(12500, 125)`
  sorted, split by the same rule as the 12,500 (90 fit / 10 validation / 25 test).
  -> data/cohorts/qm9_random125.json

Indices into QM9 refer to the 129,425-molecule QM9 of `utils.datasets.load_dataset("QM9")`.
"""

import json
import sys
from pathlib import Path

import click

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 2032025
QM9_N = 129425
SMALL_TABLE_SIZES = {"BACE": 1513, "Lipophilicity": 4200, "ESOL": 1128, "FreeSolv": 642, "QM9": 1250}


def small_sample_split(repeat):
    perm = np.random.RandomState(SEED + repeat).permutation(100)
    test, pool = perm[:40], perm[40:]
    carve = np.random.RandomState(SEED + repeat).permutation(60)
    return pool[carve[10:]], pool[carve[:10]], test  # fit, val, test


def regime_split(n, repeat, val_frac=0.1):
    pool, test = train_test_split(np.arange(n), test_size=0.2, random_state=repeat)
    perm = np.random.default_rng(repeat + 100_000).permutation(pool)
    n_val = int(round(val_frac * len(pool)))
    return perm[n_val:], perm[:n_val], test  # fit, val, test


def sorted_choice(n, k):
    return sorted(np.random.default_rng(SEED).choice(n, size=k, replace=False).tolist())


@click.command()
def main():
    """Run the checks; exit status 1 on any mismatch."""
    failures = []

    def check(name, ok):
        print(f"{'OK  ' if ok else 'FAIL'} {name}")
        if not ok:
            failures.append(name)

    small = json.loads((ROOT / "data/cohorts/small_sample.json").read_text())
    for ds, n in SMALL_TABLE_SIZES.items():
        c = small[ds]
        check(f"small-sample {ds}: 100 molecules",
              [r["table_index"] for r in c["records"]] == sorted_choice(n, 100))
        ok = True
        for s in range(15):
            fit, val, test = small_sample_split(s)
            sp = c["splits"][str(s)]
            ok &= sp["fit"] == fit.tolist() and sp["val"] == val.tolist() and sp["test"] == test.tolist()
        check(f"small-sample {ds}: 15 splits", ok)

    u1250 = [int(x) for x in (ROOT / "runners/qm9_uniform_sample_1250.txt").read_text().split()]
    check("QM9 uniform 1,250", u1250 == sorted_choice(QM9_N, 1250))
    check("small-sample QM9 rows come from the 1,250 sample",
          all(r["dataset_index"] == u1250[r["table_index"]] for r in small["QM9"]["records"]))

    r12500 = [int(x) for x in (ROOT / "data/cohorts/qm9_random12500_indices.txt").read_text().split()]
    check("QM9 random 12,500", r12500 == sorted_choice(QM9_N, 12500))

    r125 = json.loads((ROOT / "data/cohorts/qm9_random125.json").read_text())
    rows = sorted_choice(12500, 125)
    check("QM9 random 125 (rows of the 12,500)", r125["rows_in_12500"] == rows)
    check("QM9 random 125 dataset indices",
          [r["dataset_index"] for r in r125["records"]] == [r12500[i] for i in rows])
    ok = True
    for s in range(5):
        fit, val, test = regime_split(125, s)
        sp = r125["splits"][str(s)]
        ok &= sp["fit"] == fit.tolist() and sp["val"] == val.tolist() and sp["test"] == test.tolist()
    check("QM9 random 125: 5 splits", ok)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
