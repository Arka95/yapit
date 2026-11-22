"""Reuse the workspace runtime while preserving its installed backend stack."""

import argparse
import hashlib
import io
import logging
import subprocess
import sys
import tempfile
import urllib.request
import venv
import zipfile
from importlib.metadata import Distribution
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOX_URL = "https://netix.dl.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2-win32.zip"
SOX_SHA256 = "8072cc147cf1a3b3713b8b97d6844bb9389e211ab9e1101e432193fad6ae6662"
PROTECTED_PACKAGES = ("torch", "torchaudio", "numpy", "transformers", "accelerate", "ctranslate2")


def installed_stack(environment: Path) -> dict[str, str]:
    """Read installed backend versions without importing SDKs or probing hardware.

    Only the native/backend stack is protected, not the whole environment.
    """
    site = environment / "Lib" / "site-packages"
    versions = {}
    for package in PROTECTED_PACKAGES:
        for metadata in site.glob(f"{package}-*.dist-info"):
            versions[package] = Distribution.at(metadata).version
    return versions


def torch_index(mode: str, versions: dict[str, str]) -> str:
    """Keep an installed Torch build instead of switching its CPU/CUDA family.

    New GPU installations use cu128; existing CUDA installations keep their index.
    """
    version = versions.get("torch")
    build = version.partition("+")[2] if version else ""
    if build.startswith("cu") and build[2:].isdigit():
        return build
    if version and mode == "gpu":
        raise ValueError(
            "The workspace Torch build is not identified as CUDA. "
            "Keep CPU mode or explicitly resolve a compatible shared GPU stack first; "
            "setup will not replace the installed Torch build."
        )
    return "cpu" if mode == "cpu" else "cu128"


def select_mode(mode: str) -> str:
    if mode != "auto":
        return mode
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            return "gpu"
    except (OSError, subprocess.TimeoutExpired):
        pass
    logging.getLogger(__name__).warning(
        "No NVIDIA GPU detected (or driver probe unavailable); defaulting to CPU setup."
    )
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="auto", choices=("auto", "cpu", "gpu"))
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        help="Trusted offline dependency wheels; PyTorch still uses its official index.",
    )
    args = parser.parse_args()
    if sys.platform != "win32" or sys.version_info < (3, 11) or sys.maxsize <= 2**32:
        parser.error("Use 64-bit Windows Python 3.11 or newer.")
    args.mode = select_mode(args.mode)
    environment = ROOT.parent / ".venv"
    python = environment / "Scripts" / "python.exe"
    if not python.exists():
        venv.create(environment, with_pip=True, system_site_packages=False)
    config = (environment / "pyvenv.cfg").read_text()
    if "include-system-site-packages = false" not in config.lower():
        parser.error(f"{environment} must not use system site packages.")
    dependency_source = []
    if args.wheelhouse:
        wheelhouse = args.wheelhouse.resolve()
        if not wheelhouse.is_dir():
            parser.error(f"Wheel directory not found: {wheelhouse}")
        dependency_source = ["--no-index", "--find-links", str(wheelhouse)]
    versions = installed_stack(environment)
    try:
        index = torch_index(args.mode, versions)
    except ValueError as exc:
        parser.error(str(exc))
    wheels = ROOT / f"requirements-{args.mode}.txt"
    with tempfile.TemporaryDirectory(prefix="yapit-setup-") as temporary:
        constraints = Path(temporary) / "existing-stack.txt"
        constraints.write_text(
            "".join(f"{package}=={version}\n" for package, version in versions.items()),
            encoding="utf-8",
        )
        install = [str(python), "-m", "pip", "install", "-c", str(constraints)]
        subprocess.run(
            [
                *install,
                "-r",
                str(wheels),
                "--index-url",
                f"https://download.pytorch.org/whl/{index}",
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [*install, *dependency_source, "hatchling>=1.27", "editables>=0.5"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                *install,
                "--no-build-isolation",
                *dependency_source,
                "-e",
                ".[voice,rescript,mcp]",
            ],
            cwd=ROOT,
            check=True,
        )
    with urllib.request.urlopen(SOX_URL, timeout=120) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != SOX_SHA256:
        raise RuntimeError("SoX archive checksum mismatch; nothing was extracted.")
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for entry in bundle.infolist():
            name = Path(entry.filename).name
            if name == "sox.exe" or name.lower().endswith(".dll"):
                destination = environment / "Scripts" / name
                content = bundle.read(entry)
                if destination.exists() and destination.read_bytes() != content:
                    raise RuntimeError(
                        f"Refusing to overwrite a different existing tool: {destination}"
                    )
                destination.write_bytes(content)
    subprocess.run([str(environment / "Scripts" / "sox.exe"), "--version"], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    subprocess.run(
        [str(python), str(ROOT / "scripts" / "run-mode.py"), "default", "set-mode", args.mode],
        cwd=ROOT,
        check=True,
    )
    print(f"Installed {args.mode} runtime. Ready: .\\yapit.cmd doctor")


if __name__ == "__main__":
    main()
