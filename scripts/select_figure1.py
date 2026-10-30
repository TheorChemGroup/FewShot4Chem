"""Re-derive every supervised Figure 1 box from the per-repeat results and the selection rule.

For each dataset/property and each series, the box shows the 15 test scores of the candidate
configuration with the highest mean validation score over the 15 repeats. Only configurations
whose 15 validation scores are scores of the saved, tested model are eligible
(`eligible_for_validation_selection`); fine-tuning/LoRA runs that recorded only the best
training-epoch validation score are not. Series:

* best chemical FM: any chemical foundation model, frozen + head, full fine-tuning or LoRA;
* best classic baseline · non-FM head: RDKit or Morgan with linear, MLP, XGBoost or GP heads;
* best classic baseline · FM head: RDKit or Morgan with TabPFN or TabICL;
* gemini-embedding-2: Gemini embeddings with any head.

    python scripts/select_figure1.py            # print the winners
    python scripts/select_figure1.py --check    # also check figures/figure1.plotly.json and the
                                                # selected_in_figure column; exit 1 on mismatch
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
REPEATS = ROOT / "results/figure1_repeats.csv"
FIGURE = ROOT / "figures/figure1.plotly.json"
CLASSIC = {"rdkit-descriptors", "morgan-2048-r2"}
NON_FM_HEADS = {"linear", "mlp", "xgboost", "gp-rbf", "gp-matern"}
FM_HEADS = {"tabpfn", "tabicl"}


def series(r):
    if r["model"] in CLASSIC:
        if r["head"] in NON_FM_HEADS:
            return "Best classic baseline · non-FM head"
        if r["head"] in FM_HEADS:
            return "Best classic baseline · FM head"
        return None
    if r["model"] == "gemini-embedding-2":
        return "gemini-embedding-2"
    return "Best chemical FM"


def configurations():
    groups = defaultdict(list)
    for r in csv.DictReader(REPEATS.open()):
        if r["execution_status"] == "scored":
            groups[(r["dataset"], r["property"], r["model"], r["arm"], r["head"], r["rank"])].append(r)
    return groups


def winners(groups):
    best = {}
    for key, rows in sorted(groups.items()):
        s = series(rows[0])
        if s is None or len(rows) != 15 or not all(r["eligible_for_validation_selection"] == "True" for r in rows):
            continue
        mean_val = statistics.mean(float(r["val_metric"]) for r in rows)
        slot = (key[0], key[1], s)
        if slot not in best or mean_val > best[slot][0]:
            best[slot] = (mean_val, key, rows)
    return best


@click.command()
@click.option("--check", is_flag=True)
def main(check):
    groups = configurations()
    best = winners(groups)
    for (ds, prop, s), (mean_val, key, rows) in sorted(best.items()):
        test = statistics.mean(float(r["test_metric"]) for r in rows)
        print(f"{ds:13} {prop:12} {s:38} {key[2]:20} {key[3]:6} {key[4] or ('r' + key[5] if key[5] else ''):9} "
              f"val={mean_val:.4f} test={test:.4f}")
    if not check:
        return
    bad = []
    expected = {tuple(sorted(round(float(r["plotted_metric"]), 9) for r in rows)): slot
                for slot, (_, _, rows) in best.items()}
    fig = json.loads(FIGURE.read_text())
    plotted = set()
    for trace in fig["data"]:
        if trace.get("type") != "box" or not any(trace.get("name", "").startswith(p) for p in
                                                 ("Best", "gemini-embedding-2")):
            continue
        by_x = defaultdict(list)
        for x, y in zip(trace["x"], trace["y"]):
            by_x[x].append(y)
        for x, ys in by_x.items():
            slot = expected.get(tuple(sorted(round(y, 9) for y in ys)))
            if slot is None or not trace["name"].startswith(slot[2]):
                bad.append(f"box '{trace['name']}' at x={x}: not the selected configuration of its series")
            else:
                plotted.add(slot)
    for slot in sorted(set(best) - plotted):
        bad.append(f"{slot}: selected configuration not plotted")
    flagged = {k for k, rows in groups.items() if rows[0]["selected_in_figure"] == "True"}
    chosen = {key for _, key, _ in best.values()}
    if flagged != chosen:
        bad.append(f"selected_in_figure marks {len(flagged - chosen)} unselected and misses {len(chosen - flagged)} selected configurations")
    print(f"Figure 1: {len(best)} selected boxes, {len(plotted)} found in the figure, {len(bad)} problems")
    for line in bad:
        print("  MISMATCH", line)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
