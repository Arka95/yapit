"""Run YapIt in cpu or gpu mode; environment overrides are child-process-only."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "gpu"))
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="YapIt arguments, or python <args>, or mcp."
    )
    args = parser.parse_args()
    environment = ROOT / f".venv-{args.mode}"
    python = environment / "Scripts" / "python.exe"
    if not python.exists():
        parser.error(f"Missing {python}; run setup-mode.cmd {args.mode}.")
    child_env = os.environ.copy()
    child_env["YAPIT_DEVICE"] = "cuda" if args.mode == "gpu" else "cpu"
    child_env["YAPIT_ASR_DEVICE"] = child_env["YAPIT_DEVICE"]
    child_env["YAPIT_ASR_COMPUTE_TYPE"] = "float16" if args.mode == "gpu" else "int8"
    directories = [environment / "Scripts"]
    if args.mode == "gpu":
        directories.append(environment / "Lib" / "site-packages" / "torch" / "lib")
    child_env["PATH"] = (
        os.pathsep.join(map(str, directories)) + os.pathsep + child_env.get("PATH", "")
    )
    if args.mode == "gpu":
        check = subprocess.run(
            [
                str(python),
                "-c",
                "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)",
            ],
            cwd=ROOT,
            env=child_env,
        )
        if check.returncode:
            print(
                "GPU mode requires a connected NVIDIA GPU and working CUDA runtime. "
                "Use yapit-mode.cmd cpu when travelling.",
                file=sys.stderr,
            )
            return 1
    command = args.command or ["--help"]
    if command[0] == "python":
        invocation = [str(python), *command[1:]]
    elif command[0] == "mcp":
        invocation = [str(environment / "Scripts" / "yapit-mcp.exe"), *command[1:]]
    else:
        invocation = [str(environment / "Scripts" / "yapit.exe"), *command]
    if not Path(invocation[0]).is_file():
        print(f"Incomplete {args.mode} runtime; run setup-mode.cmd {args.mode}.", file=sys.stderr)
        return 1
    return subprocess.run(invocation, cwd=ROOT, env=child_env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
