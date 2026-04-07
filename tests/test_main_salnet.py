"""Smoke tests for scripts/symm_net/main_salnet.py.

Must run from scripts/symm_net/ so that `from load_utils import ...` resolves
and `../datasets` points to the cached dataset directory (scripts/datasets/).

Creates a timestamped subdir inside scripts/symm_net/runs/; a fixture cleans
that up after each test.

Default test: bp section, MNIST, 1 epoch (fast, no spiking overhead).
Slow test (--run-slow): sal section (spiking simulation adds minutes).

NOTE: first run requires MNIST to be downloaded (~10 MB). In CI, scripts/datasets/
is restored from cache so no download is needed after the first run.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import SCRIPTS_DIR

SYMM_NET_DIR = SCRIPTS_DIR / "symm_net"
SCRIPT = SYMM_NET_DIR / "main_salnet.py"
EXP_SETTINGS = Path(__file__).parent / "fixtures" / "symmnet_smoke.yaml"


@pytest.fixture
def runs_cleanup():
    """Snapshot runs/ before the test; remove any new subdirs afterward."""
    runs_dir = SYMM_NET_DIR / "runs"
    before = set(runs_dir.iterdir()) if runs_dir.exists() else set()
    yield runs_dir, before
    if runs_dir.exists():
        for new_dir in set(runs_dir.iterdir()) - before:
            shutil.rmtree(new_dir)


@pytest.mark.timeout(120)
def test_main_salnet_bp(runs_cleanup):
    runs_dir, runs_before = runs_cleanup

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "-f",
            str(EXP_SETTINGS),
            "-s",
            "bp",
            "--dataset",
            "mnist",
            "--n_epochs",
            "1",
            "--tags",
            "smoke",
        ],
        capture_output=True,
        text=True,
        cwd=str(SYMM_NET_DIR),
    )

    assert result.returncode == 0, (
        f"main_salnet.py (bp) exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    new_dirs = set(runs_dir.iterdir()) - runs_before
    assert len(new_dirs) == 1, f"expected 1 new run dir, got {len(new_dirs)}"
    run_dir = new_dirs.pop()
    assert (run_dir / "metrics.json").exists(), "metrics.json not created"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "scalars" in metrics


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_main_salnet_sal(runs_cleanup):
    runs_dir, runs_before = runs_cleanup

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "-f",
            str(EXP_SETTINGS),
            "-s",
            "sal",
            "--dataset",
            "mnist",
            "--n_epochs",
            "1",
            "--tags",
            "smoke_sal",
        ],
        capture_output=True,
        text=True,
        cwd=str(SYMM_NET_DIR),
    )

    assert result.returncode == 0, (
        f"main_salnet.py (sal) exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    new_dirs = set(runs_dir.iterdir()) - runs_before
    assert len(new_dirs) == 1, f"expected 1 new run dir, got {len(new_dirs)}"
    assert (new_dirs.pop() / "metrics.json").exists(), "metrics.json not created"
