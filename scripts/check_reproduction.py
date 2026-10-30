"""Compare freshly computed results with the published per-repeat scores.

    python scripts/check_reproduction.py heads results/smoke_heads.jsonl
    python scripts/check_reproduction.py llm results/llm_rescored.csv
    python scripts/check_reproduction.py features

`heads` / `llm`: every recomputed row is matched to its published row in
`results/figure1_repeats.csv` / `results/figure1_general_fm_repeats.csv` by
(dataset, property, model, head, repeat) and must agree to within --tol.
`features`: SHA-256 of each file in `features/small_sample/` against
`data/cohorts/feature_sha256.json` (the hashes of the feature files used for the paper).
Exit status 1 on any mismatch.
"""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
FIG1 = ROOT / "results/figure1_repeats.csv"
FIG1_LLM = ROOT / "results/figure1_general_fm_repeats.csv"
FEATURE_HASHES = ROOT / "data/cohorts/feature_sha256.json"


def report(n, bad, maxdiff, what):
    print(f"{what}: {n} compared, {len(bad)} mismatched, max |diff| = {maxdiff:.3g}")
    for line in bad[:20]:
        print("  MISMATCH", line)
    return 1 if bad or n == 0 else 0


@click.group()
def cli():
    pass


@cli.command()
@click.argument("jsonl", type=click.Path(exists=True, dir_okay=False))
@click.option("--tol", default=1e-6, show_default=True)
def heads(jsonl, tol):
    published = {}
    for r in csv.DictReader(FIG1.open()):
        if r["arm"] == "frozen" and r["execution_status"] == "scored":
            key = (r["dataset"], r["property"], r["model"], r["head"], int(float(r["repeat_number"])))
            published[key] = r
    n, bad, maxdiff, seen = 0, [], 0.0, set()
    for line in Path(jsonl).read_text().splitlines():
        row = json.loads(line)
        key = (row["dataset"], row["property"], row["model"], row["head"], int(row["repeat"]))
        pub = published.get(key)
        if pub is None:
            bad.append(f"{key}: no published row")
            continue
        if key in seen:
            bad.append(f"{key}: recomputed twice")
            continue
        seen.add(key)
        n += 1
        for mine, theirs in ((row["validation"], pub["val_metric"]), (row["test"], pub["test_metric"])):
            d = abs(float(mine) - float(theirs))
            if not math.isfinite(d):  # a NaN difference must fail, not slip past "d > tol"
                bad.append(f"{key}: non-finite {mine} vs {theirs}")
                break
            maxdiff = max(maxdiff, d)
            if d > tol:
                bad.append(f"{key}: {mine} vs {theirs}")
                break
    sys.exit(report(n, bad, maxdiff, "frozen-head scores"))


@cli.command()
@click.argument("rescored", type=click.Path(exists=True, dir_okay=False))
def llm(rescored):
    published = {}
    for r in csv.DictReader(FIG1_LLM.open()):
        key = (r["model"], r["dataset"], r["property"], int(r["repeat_number"]), r["label_condition"])
        published[key] = r
    n, bad, maxdiff = 0, [], 0.0
    for r in csv.DictReader(open(rescored)):
        key = (r["model"], r["dataset"], r["target"], int(r["seed"]), r["mode"])
        pub = published.get(key)
        if pub is None:
            bad.append(f"{key}: no published row")
            continue
        n += 1
        if r["status"] != pub["status"]:
            bad.append(f"{key}: status {r['status']} vs {pub['status']}")
        elif r["status"] == "scored":
            d = abs(float(r["value"]) - float(pub["test_metric"]))
            maxdiff = max(maxdiff, d) if math.isfinite(d) else maxdiff
            if not d <= 1e-12:
                bad.append(f"{key}: {r['value']} vs {pub['test_metric']}")
    if n != len(published):
        bad.append(f"{len(published) - n} published rows have no recomputed response")
    sys.exit(report(n, bad, maxdiff, "general-LLM scores"))


@cli.command()
def features():
    expected = json.loads(FEATURE_HASHES.read_text())
    n, bad = 0, []
    for path in sorted((ROOT / "features/small_sample").glob("*.npy")):
        want = expected.get(path.name)
        if want is None:
            continue
        n += 1
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            bad.append(f"{path.name}: {got[:16]} vs {want[:16]}")
    sys.exit(report(n, bad, 0.0, "feature files (SHA-256)"))


if __name__ == "__main__":
    cli()
