"""Provision an isolated runtime without changing the legacy environment."""

import argparse
import hashlib
import io
import subprocess
import sys
import urllib.request
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOX_URL = "https://netix.dl.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2-win32.zip"
SOX_SHA256 = "8072cc147cf1a3b3713b8b97d6844bb9389e211ab9e1101e432193fad6ae6662"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "gpu"))
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        help="Trusted offline dependency wheels; PyTorch still uses its official index.",
    )
    args = parser.parse_args()
    if sys.platform != "win32" or not (3, 11) <= sys.version_info[:2] <= (3, 12):
        parser.error("Use Windows Python 3.11 or 3.12.")
    environment = ROOT / f".venv-{args.mode}"
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
    wheels = ROOT / f"requirements-{args.mode}.txt"
    index = "cpu" if args.mode == "cpu" else "cu128"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "-r",
            str(wheels),
            "--index-url",
            f"https://download.pytorch.org/whl/{index}",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            *dependency_source,
            "hatchling>=1.27,<2",
            "editables>=0.5,<1",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            *dependency_source,
            "-c",
            str(wheels),
            "-r",
            str(ROOT / "requirements-runtime.txt"),
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
    print(f"Ready: yapit-mode.cmd {args.mode} doctor")


if __name__ == "__main__":
    main()
