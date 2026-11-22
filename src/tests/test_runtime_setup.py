"""Setup planning preserves existing backend versions without executing installers."""

import runpy
from pathlib import Path

import pytest

setup = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "setup-mode.py"))


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_setup_preserves_existing_cuda_build_in_both_execution_modes(mode):
    assert setup["torch_index"](mode, {"torch": "2.13.0+cu126"}) == "cu126"


def test_setup_rejects_replacing_an_existing_cpu_stack_with_cuda():
    with pytest.raises(ValueError, match="will not replace"):
        setup["torch_index"]("gpu", {"torch": "2.9.1+cpu"})


def test_setup_protects_backend_versions_without_freezing_unrelated_packages(tmp_path):
    for package, version in (("torch", "2.13.0+cu126"), ("numpy", "2.2.6"), ("unrelated", "1.0")):
        metadata = tmp_path / "Lib" / "site-packages" / f"{package}-{version}.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text(f"Name: {package}\nVersion: {version}\n")
    assert setup["installed_stack"](tmp_path) == {"torch": "2.13.0+cu126", "numpy": "2.2.6"}
