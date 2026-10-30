"""The entry points start, and a Figure 1 cell reproduces end to end from the raw dataset."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

ENTRY_POINTS = [
    "scripts/featurize.py",
    "scripts/fit_head.py",
    "scripts/finetune_small_sample.py",
    "scripts/qm9_curve.py",
    "scripts/query_llm.py",
    "scripts/score_llm_responses.py",
    "runners/finetune_one.py",
    "runners/run_one_eval.py",
]


def run(*args, timeout=1800):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout, check=False)


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_entry_point_starts(script):
    result = run(script, "--help", timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.slow
def test_esol_rdkit_linear_head_reproduces(tmp_path):
    """Downloads ESOL (~100 kB) on first use."""
    featurized = run("scripts/featurize.py", "--dataset", "ESOL", "--model-config",
                     "models/configs/descriptors/rdkit.yaml")
    assert featurized.returncode == 0, featurized.stderr[-2000:]
    assert run("scripts/check_reproduction.py", "features").returncode == 0
    out = tmp_path / "heads.jsonl"
    fitted = run("scripts/fit_head.py", "--dataset", "ESOL", "--model", "rdkit-descriptors",
                 "--head", "linear", "--repeats", "0-2", "--out", str(out))
    assert fitted.returncode == 0, fitted.stderr[-2000:]
    checked = run("scripts/check_reproduction.py", "heads", str(out))
    assert checked.returncode == 0, checked.stdout
