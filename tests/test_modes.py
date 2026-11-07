import importlib.util
import json
import os
import shutil
import subprocess
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


@pytest.mark.skipif(sys.platform != "win32" or not shutil.which("py"), reason="Windows py launcher")
@pytest.mark.parametrize(
    ("wrapper", "arguments", "backend", "expected"),
    [
        ("setup-mode.cmd", ["--help"], "setup-mode.py", ["--help"]),
        ("yapit-mode.cmd", ["cpu", "--help"], "run-mode.py", ["cpu", "--help"]),
        ("Start Enrollment.cmd", [], "run-mode.py", ["cpu", "enroll-app"]),
        ("Start Enrollment.cmd", ["gpu"], "run-mode.py", ["gpu", "enroll-app"]),
    ],
)
def test_shell_launchers_resolve_moved_backends_and_preserve_exit_codes(
    tmp_path, wrapper, arguments, backend, expected
):
    scripts = tmp_path / "project with spaces" / "scripts"
    scripts.mkdir(parents=True)
    original = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("setup-mode.cmd", "yapit-mode.cmd", "Start Enrollment.cmd"):
        shutil.copyfile(original / name, scripts / name)
    stub = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "print(json.dumps({'args': sys.argv[1:], 'backend': Path(__file__).name}))\n"
        "raise SystemExit(7)\n"
    )
    for name in ("setup-mode.py", "run-mode.py"):
        (scripts / name).write_text(stub)
    result = subprocess.run(
        f'call "{scripts / wrapper}" {subprocess.list2cmdline(arguments)}',
        shell=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 7, result.stderr
    assert json.loads(result.stdout) == {"args": expected, "backend": backend}


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
