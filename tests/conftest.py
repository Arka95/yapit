import os
import subprocess
import sys

import pytest


@pytest.fixture(autouse=True)
def isolated_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("YAPIT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("YAPIT_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("YAPIT_MODELS_DIR", str(tmp_path / "models"))


@pytest.fixture
def run_yapit(tmp_path):
    """Exercise the installed command as a separate process, never an in-process API."""

    def run(*arguments):
        return subprocess.run(
            [sys.executable, "-m", "yapit.cli", *arguments],
            cwd=tmp_path,
            env={**os.environ, "YAPIT_ALLOW_CLOUD": "false"},
            capture_output=True,
            text=True,
            timeout=30,
        )

    return run
