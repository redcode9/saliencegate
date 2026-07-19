from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures"

RunCli = Callable[..., subprocess.CompletedProcess[str]]


@pytest.fixture
def run_cli(tmp_path: Path) -> RunCli:
    home = tmp_path / "home"
    home.mkdir()

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            HOME=str(home),
            PYTHONUTF8="1",
            SALIENCEGATE_TESTING="1",
        )
        return subprocess.run(
            (sys.executable, "-m", "saliencegate", *arguments),
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    return run
