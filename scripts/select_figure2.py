"""Recompute the Figure 2 points (QM9 ε_LUMO learning curve) from all candidate repeat scores.

For each support size (100, 1,000, 10,000 training + validation molecules), model and adaptation
arm, the plotted point is the mean test R² over five repeats of the configuration (head or LoRA
rank) with the highest mean validation R² among configurations scored on all five repeats
(ties broken by configuration name). Input: `results/figure2_candidate_repeats.csv`.

The support-60 points come from the Figure 1 small-sample runs (`results/figure1_*`). The GPT-6
astra point at 1,000 (no configuration to select) is re-scored here from its stored predictions
(`results/llm_uniform1250/`) against `data/llm_table_probe/qm9_uniform1250.csv`.

    python scripts/select_figure2.py --check results/figure2_points.csv
"""

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "results/figure2_candidate_repeats.csv"
# Figure 2 model names -> candidate-table names
PANEL_ARM = {"frozen": "frozen", "fullft": "fullft", "lora": "lora"}


def astra_uniform1250():
    """Mean test R² of the five GPT-6 astra runs with 1,000 labelled molecules in the prompt."""
    truth = [float(r["epsilon_lumo"]) for r in csv.DictReader((ROOT / "data/llm_table_probe/qm9_uniform1250.csv").open())]
    scores = []
    for path in sorted((ROOT / "results/llm_uniform1250").glob("*.json")):
        run = json.loads(path.read_text())
        ids = list(run["predictions"])
        y = [truth[int(run["query_idx"][i])] for i in ids]
        p = [float(run["predictions"][i]) for i in ids]
        my = sum(y) / len(y)
        scores.append(1 - sum((a - b) ** 2 for a, b in zip(y, p)) / sum((a - my) ** 2 for a in y))
    assert len(scores) == 5, scores
    return statistics.mean(scores)


def select():
    groups = defaultdict(dict)
    for r in csv.DictReader(CANDIDATES.open()):
        key = (int(r["support"]), r["model"], r["arm"], r["configuration"])
        assert int(r["repeat"]) not in groups[key], key
        groups[key][int(r["repeat"])] = r
    best = {}
    for (support, model, arm, config), reps in groups.items():
        if set(reps) != set(range(5)):
            continue
        val = statistics.mean(float(reps[i]["validation_R2"]) for i in range(5))
        test = statistics.mean(float(reps[i]["test_R2"]) for i in range(5))
        cand = (-val, config, test)
        k = (support, model, arm)
        if k not in best or cand < best[k]:
            best[k] = cand
    return {k: {"configuration": c[1], "mean_validation_R2": -c[0], "mean_test_R2": c[2]} for k, c in best.items()}


@click.command()
@click.option("--check", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Published points CSV to compare with.")
def main(check):
    chosen = select()
    for (support, model, arm), c in sorted(chosen.items()):
        print(f"{support:>6} {arm:7} {model:22} {c['configuration']:10} test={c['mean_test_R2']:.4f}")
    if not check:
        return
    n, bad = 0, []
    for p in csv.DictReader(open(check)):
        support = int(float(p["support"]))
        if support == 60 or p["status"] != "plotted":
            continue
        model = {"gin-pretrained": "gin-pretrained"}.get(p["model"], p["model"])
        arm = "frozen" if model in ("rdkit", "gemini") else PANEL_ARM[p["panel"]]
        if model == "astra":
            want = astra_uniform1250() if support == 1000 else None
            if want is None or not math.isclose(want, float(p["plotted_R2"]), abs_tol=1e-9):
                bad.append(f"{p['panel']} astra {support}: {want} vs {p['plotted_R2']}")
            n += 1
            continue
        c = chosen.get((support, model, arm))
        if c is None:
            bad.append(f"{p['panel']} {model} {support}: nothing selected")
            continue
        n += 1
        if c["configuration"] != p["configuration"] or not math.isclose(
            c["mean_test_R2"], float(p["plotted_R2"]), abs_tol=1e-9
        ):
            bad.append(f"{p['panel']} {model} {support}: {c['configuration']} {c['mean_test_R2']} "
                       f"vs {p['configuration']} {p['plotted_R2']}")
    print(f"Figure 2 points: {n} recomputed, {len(bad)} mismatched")
    for line in bad:
        print("  MISMATCH", line)
    sys.exit(1 if bad or n == 0 else 0)


if __name__ == "__main__":
    main()
