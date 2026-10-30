"""Score general-purpose LLM table-prompting responses (Figure 1 "general-purpose FMs").

Each task prompts a model with 60 labelled support molecules (SMILES<TAB>value) and asks for
predictions on 40 query molecules; `mode=shuffled` permutes the support labels as a control.
Tasks (prompts and query truth) are in `data/llm_table_probe/tasks_*.json`; the raw API
responses are in `results/llm_responses/` (one JSON per model × dataset × target × repeat × mode).

A response counts as scored only if it is not a refusal and returns exactly one finite
prediction per query id (probabilities in [0, 1] for BACE). Refusals and invalid outputs are
reported as such and never scored as zero.

    python scripts/score_llm_responses.py --out results/llm_rescored.csv
"""

import csv
import json
import math
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]


TASKS = sorted((ROOT / "data/llm_table_probe").glob("tasks_*.json"))
RESPONSES = ROOT / "results/llm_responses"
FIELDS = ["model", "dataset", "target", "seed", "mode", "status", "metric", "value", "finish_reason"]


def r2(y, p):
    """1 - SS_res / SS_tot around the mean of y (equals sklearn's r2_score)."""
    my = sum(y) / len(y)
    return 1.0 - sum((a - b) ** 2 for a, b in zip(y, p)) / sum((a - my) ** 2 for a in y)


def roc_auc(y, s):
    """ROC-AUC by the rank-sum formula, ties sharing the average rank (equals sklearn's)."""
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    pos = [i for i, v in enumerate(y) if v == 1]
    neg = len(y) - len(pos)
    return (sum(ranks[i] for i in pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * neg)


def parse(response, task):
    """Same acceptance rule as the run that produced the paper's numbers."""
    choice = (response.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    fr, native = choice.get("finish_reason"), choice.get("native_finish_reason")
    if msg.get("refusal") or fr == "content_filter" or native == "refusal":
        return {"status": "refused", "finish_reason": fr}
    try:
        arr = json.loads(msg.get("content") or "")["predictions"]
        ids = [x["id"] for x in arr]
        assert len(ids) == len(set(ids)) == task["n_query"]
        assert set(ids) == set(task["truth"])
        pred = {x["id"]: float(x["value"]) for x in arr}
        assert all(math.isfinite(v) for v in pred.values())
        if task["dataset"] == "BACE":
            assert all(0 <= v <= 1 for v in pred.values())
    except (ValueError, KeyError, TypeError, AssertionError):
        return {"status": "output_capped" if fr == "length" else "incomplete_invalid", "finish_reason": fr}
    ids = list(task["truth"])
    y, p = [task["truth"][i] for i in ids], [pred[i] for i in ids]
    bace = task["dataset"] == "BACE"
    return {
        "status": "scored",
        "metric": "ROC_AUC" if bace else "R2",
        "value": roc_auc(y, p) if bace else r2(y, p),
        "finish_reason": fr,
    }


@click.command()
@click.option("--responses", type=click.Path(file_okay=False, exists=True), default=str(RESPONSES), show_default=True)
@click.option("--out", type=click.Path(dir_okay=False), default=str(ROOT / "results/llm_rescored.csv"))
def main(responses, out):
    tasks = {}
    for path in TASKS:
        for t in json.loads(path.read_text()):
            tasks[(t["dataset"], t["target"], int(t["seed"]), t["mode"])] = t
    rows = []
    for path in sorted(Path(responses).glob("*.json")):
        if path.name.endswith(".usage.json"):
            continue
        provider, name, dataset, target, seed, mode = path.stem.split("__")
        task = tasks[(dataset, target, int(seed), mode)]
        row = {"model": f"{provider}/{name}", "dataset": dataset, "target": target, "seed": int(seed), "mode": mode}
        row.update(parse(json.loads(path.read_text()), task))
        rows.append(row)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    scored = sum(r["status"] == "scored" for r in rows)
    print(f"{len(rows)} responses, {scored} scored -> {out}")


if __name__ == "__main__":
    main()
