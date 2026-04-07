"""Smoke tests for scripts/symm_net/main_salnet.py.

Must run from scripts/symm_net/ so that `from load_utils import ...` resolves
and `../datasets` points to the cached dataset directory (scripts/datasets/).

Results are written to --output-dir, which the tests point at tmp_path so
pytest handles cleanup automatically.

Default test: bp section, MNIST, 1 epoch (fast, no spiking overhead).
Slow test (--run-slow): sal section (spiking simulation adds minutes).

NOTE: first run requires MNIST to be downloaded (~10 MB). In CI, scripts/datasets/
is restored from cache so no download is needed after the first run.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import SCRIPTS_DIR

SYMM_NET_DIR = SCRIPTS_DIR / "symm_net"
SCRIPT = SYMM_NET_DIR / "main_salnet.py"
EXP_SETTINGS = Path(__file__).parent / "fixtures" / "symmnet_smoke.yaml"


@pytest.mark.timeout(120)
def test_main_salnet_bp(tmp_path):
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
            "--output-dir",
            str(tmp_path),
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

    metrics_files = list(tmp_path.rglob("metrics.json"))
    assert len(metrics_files) == 1, "metrics.json not found"
    metrics = json.loads(metrics_files[0].read_text())
    assert "scalars" in metrics


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_main_salnet_sal(tmp_path):
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
            "--output-dir",
            str(tmp_path),
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

    assert list(tmp_path.rglob("metrics.json")), "metrics.json not found"
