"""Check that the figures plot exactly the numbers in the results tables.

    python scripts/check_figures.py

Figure 1: every box (one per dataset and series) must contain exactly the per-repeat plotted
scores of one configuration in `results/figure1_repeats.csv` (rows marked `selected_in_figure`)
or of one model × label condition in `results/figure1_general_fm_repeats.csv`.
Figure 2 / S2: every plotted point must equal `plotted_R2` in `results/figure2_points.csv`.
Exit status 1 on any mismatch.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


def key(values):
    return tuple(sorted(round(v, 9) for v in values))


def figure1():
    groups = defaultdict(list)
    for r in csv.DictReader((RESULTS / "figure1_repeats.csv").open()):
        if r["selected_in_figure"] == "True" and r["plotted_metric"]:
            groups[("supervised", r["dataset"], r["property"], r["model"], r["arm"], r["head"], r["rank"])].append(float(r["plotted_metric"]))
    for r in csv.DictReader((RESULTS / "figure1_general_fm_repeats.csv").open()):
        if r["plotted_metric"]:
            groups[("llm", r["dataset"], r["property"], r["model"], r["label_condition"])].append(float(r["plotted_metric"]))
    known = {key(v): k for k, v in groups.items()}
    fig = json.loads((FIGURES / "figure1.plotly.json").read_text())
    n, bad = 0, []
    for trace in fig["data"]:
        if trace.get("type") != "box":
            continue
        by_x = defaultdict(list)
        for x, y in zip(trace["x"], trace["y"]):
            if y is not None:
                by_x[x].append(y)
        for x, ys in sorted(by_x.items()):
            n += 1
            if key(ys) not in known:
                bad.append(f"Figure 1 box '{trace.get('name')}' at x={x}: {len(ys)} values match no table group")
    return n, bad


def figure2(name):
    points = {}
    for p in csv.DictReader((RESULTS / "figure2_points.csv").open()):
        if p["status"] == "plotted":
            points[(p["panel"], p["model"], int(float(p["support"])))] = float(p["plotted_R2"])
    fig = json.loads((FIGURES / f"{name}.plotly.json").read_text())
    n, bad = 0, []
    for trace in fig["data"]:
        meta = trace.get("meta") or {}
        if meta.get("role") in (None, "boundary_indicator"):
            continue
        model = meta.get("model") if meta["role"] == "chemical" else {"rdkit": "rdkit", "embedding": "gemini", "astra": "astra"}[meta["role"]]
        for x, y in zip(trace["x"], trace["y"]):
            if y is None:
                continue
            n += 1
            want = points.get((meta["arm"], model, int(x)))
            if want is None or not math.isclose(y, want, abs_tol=1e-9):
                bad.append(f"{name} {meta['arm']} {model} x={x}: plotted {y}, table {want}")
    return n, bad


@click.command()
def main():
    """Run the checks; exit status 1 on any mismatch."""
    status = 0
    for label, (n, bad) in (("Figure 1 boxes", figure1()), ("Figure 2 points", figure2("figure2")),
                            ("Figure S2 points", figure2("figureS2"))):
        print(f"{label}: {n} checked, {len(bad)} mismatched")
        for line in bad[:20]:
            print("  MISMATCH", line)
        status |= bool(bad) or n == 0
    sys.exit(int(status))


if __name__ == "__main__":
    main()
