"""Runtime selection uses prepared shared environments without hardware probes."""

from pathlib import Path

import pytest

from yapit.common_utils.runtime_config import runtime_environment


def shared_runtime(root: Path, version: str | None, *, installed: bool = True) -> Path:
    environment = root.parent / ".venv"
    if installed:
        executable = environment / "Scripts" / "yapit.exe"
        executable.parent.mkdir(parents=True)
        executable.touch()
    if version:
        metadata = environment / "Lib" / "site-packages" / f"torch-{version}.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text(f"Name: torch\nVersion: {version}\n")
    return environment


@pytest.mark.parametrize(
    "mode,version", [("cpu", None), ("cpu", "2.9.1+cu128"), ("gpu", "2.13.0+cu126")]
)
def test_prepared_compatible_workspace_runtime_is_reused(tmp_path, mode, version):
    root = tmp_path / "project"
    environment = shared_runtime(root, version)
    assert runtime_environment(root, mode) == environment


@pytest.mark.parametrize(
    "installed,version", [(False, "2.9.1+cu128"), (True, None), (True, "2.9.1+cpu")]
)
def test_incomplete_or_cpu_only_workspace_keeps_existing_gpu_runtime(tmp_path, installed, version):
    root = tmp_path / "project"
    shared_runtime(root, version, installed=installed)
    assert runtime_environment(root, "gpu") == root / ".venv-gpu"
