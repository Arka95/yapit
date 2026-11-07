import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run-mode.py"
    spec = importlib.util.spec_from_file_location("mode_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_mode_overrides_shared_config_without_mutating_parent(monkeypatch, tmp_path):
    module = launcher()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    python = tmp_path / ".venv-cpu" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    (python.parent / "yapit.exe").touch()
    monkeypatch.setenv("YAPIT_DEVICE", "cuda")
    monkeypatch.setenv("YAPIT_ASR_DEVICE", "cuda")
    monkeypatch.setenv("YAPIT_WORKSPACE", r"D:\shared\workspace")
    monkeypatch.setenv("YAPIT_DATA_DIR", r"D:\shared\data")
    original_path = os.environ["PATH"]
    monkeypatch.setattr(sys, "argv", ["run-mode.py", "cpu", "doctor"])
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(returncode=0),
    )
    assert module.main() == 0
    env = calls[0][1]["env"]
    assert env["YAPIT_DEVICE"] == "cpu"
    assert env["YAPIT_ASR_DEVICE"] == "cpu"
    assert env["YAPIT_ASR_COMPUTE_TYPE"] == "int8"
    assert env["YAPIT_WORKSPACE"] == r"D:\shared\workspace"
    assert env["YAPIT_DATA_DIR"] == r"D:\shared\data"
    assert os.environ["YAPIT_DEVICE"] == "cuda"
    assert os.environ["PATH"] == original_path


def test_gpu_missing_device_stops_before_dispatch(monkeypatch, tmp_path, capsys):
    module = launcher()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    python = tmp_path / ".venv-gpu" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(sys, "argv", ["run-mode.py", "gpu", "doctor"])
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(returncode=1),
    )
    assert module.main() == 1
    assert len(calls) == 1
    assert calls[0][1]["env"]["YAPIT_DEVICE"] == "cuda"
    assert "Use yapit-mode.cmd cpu" in capsys.readouterr().err


def test_missing_mode_environment_is_actionable(monkeypatch, tmp_path, capsys):
    module = launcher()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run-mode.py", "cpu", "doctor"])
    with pytest.raises(SystemExit):
        module.main()
    assert "setup-mode.cmd cpu" in capsys.readouterr().err
