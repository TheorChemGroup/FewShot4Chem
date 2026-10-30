"""The shipped tables, figures and cohorts are reproducible from the shipped data (fast, CPU)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import xlsx  # noqa: E402


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, check=False)


def test_cohorts_follow_their_sampling_rules():
    result = run("scripts/make_cohorts.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout


def test_tables_rebuild_cell_for_cell(tmp_path):
    result = run("scripts/build_tables.py", "--out-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    for name in ("Figure1_reproduction", "Figure1_general_FMs", "Figure2_reproduction"):
        assert xlsx.read(tmp_path / f"{name}.xlsx") == xlsx.read(ROOT / "tables" / f"{name}.xlsx"), name


def test_llm_scores_reproduce_from_stored_responses(tmp_path):
    out = tmp_path / "rescored.csv"
    assert run("scripts/score_llm_responses.py", "--out", str(out)).returncode == 0
    result = run("scripts/check_reproduction.py", "llm", str(out))
    assert result.returncode == 0, result.stdout


def test_figure2_points_follow_the_selection_rule():
    result = run("scripts/select_figure2.py", "--check", "results/figure2_points.csv")
    assert result.returncode == 0, result.stdout


def test_figures_plot_the_table_values():
    result = run("scripts/check_figures.py")
    assert result.returncode == 0, result.stdout


def test_figure1_boxes_follow_the_selection_rule():
    result = run("scripts/select_figure1.py", "--check")
    assert result.returncode == 0, result.stdout
