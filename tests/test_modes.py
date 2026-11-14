import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from dotenv import dotenv_values


def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run-mode.py"
    spec = importlib.util.spec_from_file_location("mode_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_launcher():
    path = Path(__file__).resolve().parents[1] / "scripts" / "setup-mode.py"
    spec = importlib.util.spec_from_file_location("setup_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "probe",
    [
        SimpleNamespace(returncode=0, stdout=""),
        SimpleNamespace(returncode=1, stdout=""),
        FileNotFoundError("nvidia-smi"),
        subprocess.TimeoutExpired("nvidia-smi", 10),
    ],
)
def test_automatic_setup_defaults_to_cpu_with_a_visible_reason(monkeypatch, caplog, probe):
    module = setup_launcher()

    def run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        assert kwargs["timeout"] == 10
        if isinstance(probe, Exception):
            raise probe
        return probe

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.select_mode("auto") == "cpu"
    assert "defaulting to CPU setup" in caplog.text


def test_automatic_setup_selects_detected_nvidia_gpu(monkeypatch):
    module = setup_launcher()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="Test NVIDIA GPU\n"),
    )
    assert module.select_mode("auto") == "gpu"


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_explicit_setup_modes_do_not_probe_or_fall_back(monkeypatch, mode):
    module = setup_launcher()

    def unexpected_probe(*args, **kwargs):
        pytest.fail("Explicit setup must preserve the requested mode without probing.")

    monkeypatch.setattr(module.subprocess, "run", unexpected_probe)
    assert module.select_mode(mode) == mode


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
    assert "Use scripts\\yapit-mode.cmd cpu" in capsys.readouterr().err


def test_mode_launcher_dispatches_mcp_module_without_a_stale_console_entry(monkeypatch, tmp_path):
    module = launcher()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    python = tmp_path / ".venv-cpu" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(sys, "argv", ["run-mode.py", "cpu", "mcp"])
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(returncode=0),
    )
    assert module.main() == 0
    assert calls[0][0] == [str(python), "-m", "yapit.mcp"]
    assert calls[0][1]["env"]["YAPIT_DEVICE"] == "cpu"


def test_missing_mode_environment_is_actionable(monkeypatch, tmp_path, capsys):
    module = launcher()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run-mode.py", "cpu", "doctor"])
    with pytest.raises(SystemExit):
        module.main()
    assert "scripts\\setup-mode.cmd cpu" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_set_mode_persists_choice_and_next_call_uses_it(monkeypatch, tmp_path, capsys, mode):
    module = launcher()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.delenv("YAPIT_MODE", raising=False)
    config = tmp_path / ".env"
    config.write_text("# Existing settings\nYAPIT_DATA_DIR=personal-data\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run-mode.py", "default", "set-mode", mode])
    assert module.main() == 0
    settings = json.loads(capsys.readouterr().out)
    assert settings["mode"] == mode and not settings["environment_override"]
    assert dotenv_values(config)["YAPIT_DATA_DIR"] == "personal-data"
    assert "YAPIT_MODE" not in os.environ

    scripts = tmp_path / f".venv-{mode}" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").touch()
    (scripts / "yapit.exe").touch()
    monkeypatch.setattr(sys, "argv", ["run-mode.py", "default", "prompt", "--topic", "A calm trip"])
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0 if command[0].endswith("python.exe") else 7)

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.main() == 7
    assert calls[-1][0] == [str(scripts / "yapit.exe"), "prompt", "--topic", "A calm trip"]
    assert calls[-1][1]["env"]["YAPIT_MODE"] == mode
    assert calls[-1][1]["env"]["YAPIT_DEVICE"] == ("cuda" if mode == "gpu" else "cpu")


def test_process_mode_overrides_saved_mode_without_changing_it(monkeypatch, tmp_path):
    module = launcher()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    config = tmp_path / ".env"
    config.write_text("YAPIT_MODE=gpu\n", encoding="utf-8")
    monkeypatch.setenv("YAPIT_MODE", "cpu")
    scripts = tmp_path / ".venv-cpu" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").touch()
    (scripts / "yapit.exe").touch()
    monkeypatch.setattr(sys, "argv", ["run-mode.py", "default", "doctor"])
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )
    assert module.main() == 0
    assert calls == [[str(scripts / "yapit.exe"), "doctor"]]
    assert dotenv_values(config)["YAPIT_MODE"] == "gpu"
    assert os.environ["YAPIT_MODE"] == "cpu"


@pytest.mark.parametrize("source", ["environment", "config", "set-mode"])
def test_invalid_mode_is_rejected_without_changing_configuration(
    monkeypatch, tmp_path, capsys, source
):
    module = launcher()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.delenv("YAPIT_MODE", raising=False)
    config = tmp_path / ".env"
    contents = "YAPIT_MODE=unknown\n" if source == "config" else "YAPIT_MODE=cpu\n"
    config.write_text(contents, encoding="utf-8")
    if source == "environment":
        monkeypatch.setenv("YAPIT_MODE", "unknown")
    command = ["set-mode", "unknown"] if source == "set-mode" else ["doctor"]
    monkeypatch.setattr(sys, "argv", ["run-mode.py", "default", *command])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert "cpu" in capsys.readouterr().err
    assert config.read_text(encoding="utf-8") == contents


def test_saved_gpu_mode_does_not_fall_back_to_an_installed_cpu(monkeypatch, tmp_path, capsys):
    module = launcher()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.delenv("YAPIT_MODE", raising=False)
    (tmp_path / ".env").write_text("YAPIT_MODE=gpu\n", encoding="utf-8")
    for mode in ("cpu", "gpu"):
        scripts = tmp_path / f".venv-{mode}" / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "python.exe").touch()
        (scripts / "yapit.exe").touch()
    monkeypatch.setattr(sys, "argv", ["run-mode.py", "default", "doctor"])
    calls = []

    def unavailable(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(module.subprocess, "run", unavailable)
    assert module.main() == 1
    assert len(calls) == 1 and ".venv-gpu" in calls[0][0]
    assert "set-mode cpu" in capsys.readouterr().err


def test_gpu_help_does_not_probe_hardware(monkeypatch, tmp_path):
    module = launcher()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    scripts = tmp_path / ".venv-gpu" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").touch()
    (scripts / "yapit.exe").touch()
    monkeypatch.setattr(sys, "argv", ["run-mode.py", "gpu", "--help"])
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )
    assert module.main() == 0
    assert calls == [[str(scripts / "yapit.exe"), "--help"]]


def test_set_mode_cli_saves_local_setting_without_enrollment_or_models(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from yapit.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("YAPIT_MODE", raising=False)
    result = CliRunner().invoke(app, ["set-mode", "gpu"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["mode"] == "gpu"
    assert dotenv_values(tmp_path / ".env")["YAPIT_MODE"] == "gpu"
