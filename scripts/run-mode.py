"""Run YapIt in an isolated runtime; environment overrides are child-process-only."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from yapit.common_utils.runtime_config import RUNTIME_MODES, runtime_mode, set_runtime_mode

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("default", *RUNTIME_MODES))
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="YapIt arguments, or python <args>, or mcp."
    )
    args = parser.parse_args()
    command = args.command or ["--help"]
    if command[0] == "set-mode":
        settings_parser = argparse.ArgumentParser(prog="yapit.cmd set-mode")
        settings_parser.add_argument("mode", choices=RUNTIME_MODES)
        settings = settings_parser.parse_args(command[1:])
        print(json.dumps(set_runtime_mode(ROOT, settings.mode), indent=2))
        return 0
    try:
        mode = runtime_mode(ROOT) if args.mode == "default" else args.mode
    except ValueError as exc:
        parser.error(str(exc))
    environment = ROOT / f".venv-{mode}"
    python = environment / "Scripts" / "python.exe"
    if not python.is_file():
        parser.error(f"Missing {python}; run scripts\\setup-mode.cmd {mode}.")
    child_env = os.environ.copy()
    child_env["YAPIT_MODE"] = mode
    child_env["YAPIT_DEVICE"] = "cuda" if mode == "gpu" else "cpu"
    child_env["YAPIT_ASR_DEVICE"] = child_env["YAPIT_DEVICE"]
    child_env["YAPIT_ASR_COMPUTE_TYPE"] = "float16" if mode == "gpu" else "int8"
    directories = [environment / "Scripts"]
    if mode == "gpu":
        directories.append(environment / "Lib" / "site-packages" / "torch" / "lib")
    child_env["PATH"] = (
        os.pathsep.join(map(str, directories)) + os.pathsep + child_env.get("PATH", "")
    )
    if mode == "gpu" and "--help" not in command:
        try:
            check = subprocess.run(
                [
                    str(python),
                    "-c",
                    "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)",
                ],
                cwd=ROOT,
                env=child_env,
                capture_output=True,
                timeout=10,
                check=False,
            )
            ready = check.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ready = False
        if not ready:
            print(
                "GPU mode requires a connected NVIDIA GPU and working CUDA runtime. "
                "Use yapit.cmd set-mode cpu to switch (install its runtime if needed). "
                "Use scripts\\yapit-mode.cmd cpu for a one-off override.",
                file=sys.stderr,
            )
            return 1
    if command[0] == "python":
        invocation = [str(python), *command[1:]]
    elif command[0] == "mcp":
        invocation = [str(python), "-m", "yapit.mcp", *command[1:]]
    else:
        invocation = [str(environment / "Scripts" / "yapit.exe"), *command]
    if not Path(invocation[0]).is_file():
        print(
            f"Incomplete {mode} runtime; run scripts\\setup-mode.cmd {mode}.",
            file=sys.stderr,
        )
        return 1
    return subprocess.run(invocation, cwd=ROOT, env=child_env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
