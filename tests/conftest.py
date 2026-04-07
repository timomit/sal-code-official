from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (SAL/RDD spiking simulations).",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(
            reason="slow test — run with --run-slow to include"
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
