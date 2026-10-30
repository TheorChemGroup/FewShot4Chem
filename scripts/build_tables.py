"""Build the reproduction tables (`tables/*.xlsx`) from the per-repeat results in `results/`.

    python scripts/build_tables.py

* `tables/Figure1_reproduction.xlsx` — every Figure 1 candidate configuration of the chemical
  FMs, RDKit, Morgan and Gemini embeddings, one row per repeat (15 repeats): validation and test
  score. ROC-AUC for BACE, R² otherwise. Failed fits keep blank scores, never zero.
* `tables/Figure1_general_FMs.xlsx` — table-prompted general-purpose LLMs, one row per request.
* `tables/Figure2_reproduction.xlsx` — the plotted Figure 2 points (sheet `Points`) and their
  per-repeat scores (sheet `Repeat scores`).

Scores keep full precision; Excel displays up to four decimals.
"""

import csv
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xlsx  # noqa: E402

RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
QM9_LABEL = {"epsilon_lumo": "εLUMO", "U_0": "U₀"}


def number(v):
    return float(v) if v not in ("", None) else None


def integer(v):
    return int(float(v)) if v not in ("", None) else None


def dataset_label(r):
    if r["dataset"] == "QM9":
        return "QM9 " + QM9_LABEL[r["property"]]
    return "Lipo" if r["dataset"] == "Lipophilicity" else r["dataset"]


def adaptation_label(r):
    rank = integer(r["rank"])
    return f"{r['adaptation_method']} (r={rank})" if rank else r["adaptation_method"]


def figure1_rows():
    header = ["Dataset", "model", "adaptation_method", "head", "repeat_number", "val_metric", "test_metric"]
    rows = [
        [dataset_label(r), r["model"], adaptation_label(r), r["head"], integer(r["repeat_number"]),
         number(r["val_metric"]), number(r["test_metric"])]
        for r in csv.DictReader((RESULTS / "figure1_repeats.csv").open())
    ]
    return header, rows


def general_fm_rows():
    header = ["Dataset", "model", "labels", "metric", "repeat_number", "status", "test_metric"]
    rows = [
        [dataset_label(r), r["model"], r["label_condition"], r["metric"], integer(r["repeat_number"]),
         r["status"], number(r["test_metric"])]
        for r in csv.DictReader((RESULTS / "figure1_general_fm_repeats.csv").open())
    ]
    return header, rows


def figure2_sheets():
    numeric_point = {"support", "test_repeats", "mean_validation_R2", "mean_test_R2", "plotted_R2"}
    numeric_repeat = {"support", "repeat", "validation_R2", "test_R2"}
    sheets, number_columns = [], {}
    for name, filename, numeric in (("Points", "figure2_points.csv", numeric_point),
                                    ("Repeat scores", "figure2_repeat_scores.csv", numeric_repeat)):
        reader = csv.DictReader((RESULTS / filename).open())
        header = reader.fieldnames
        rows = [[(number(r[k]) if k in numeric else r[k]) for k in header] for r in reader]
        sheets.append((name, header, rows))
        number_columns[name] = {j for j, k in enumerate(header) if k in numeric - {"support", "repeat", "test_repeats"}}
    return sheets, number_columns


@click.command()
@click.option("--out-dir", type=click.Path(file_okay=False), default=str(TABLES), show_default=True)
def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    header, rows = figure1_rows()
    xlsx.write(out / "Figure1_reproduction.xlsx", [("Repeat scores", header, rows)], {"Repeat scores": {5, 6}})
    header, rows = general_fm_rows()
    xlsx.write(out / "Figure1_general_FMs.xlsx", [("Repeat scores", header, rows)], {"Repeat scores": {6}})
    sheets, number_columns = figure2_sheets()
    xlsx.write(out / "Figure2_reproduction.xlsx", sheets, number_columns)
    for name in ("Figure1_reproduction", "Figure1_general_FMs", "Figure2_reproduction"):
        book = xlsx.read(out / f"{name}.xlsx")
        print(name, {k: len(v) - 1 for k, v in book.items()})


if __name__ == "__main__":
    main()
