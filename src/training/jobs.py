"""Submit explicit desktop training jobs to the configured Linux runtime."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

from yapit.common_utils.metrics import timed
from yapit.constants import models_directory
from yapit.storage import Workspace
from yapit.training.trainer import train


def _wsl_paths(command: list[str], paths: list[str], *, windows: bool = False) -> list[str]:
    flag = "-w" if windows else "-u"
    result = subprocess.run(
        [
            *command,
            "sh",
            "-c",
            f'for path; do wslpath -a {flag} "$path" || exit; done',
            "yapit-paths",
            *paths,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    converted = result.stdout.splitlines()
    if len(converted) != len(paths) or not all(converted):
        raise RuntimeError("The configured WSL runtime could not translate training paths.")
    return converted


@timed
def run_training(
    workspace: Workspace,
    name: str,
    upstream: Path,
    *,
    epochs: int,
    batch_size: int,
    model: str,
) -> dict:
    """Train locally on Linux, or use an explicitly configured WSL Python on Windows."""
    if platform.system() != "Windows":
        return train(workspace, name, upstream, epochs=epochs, batch_size=batch_size, model=model)
    python = os.environ.get("YAPIT_TRAINING_PYTHON", "").strip()
    if not python.startswith("/"):
        raise RuntimeError(
            "Training needs a configured Linux/WSL2 environment with YapIt, CUDA and "
            "FlashAttention. Set YAPIT_TRAINING_PYTHON to its Python executable; "
            "no training environment is installed automatically."
        )
    command = ["wsl.exe"]
    distribution = os.environ.get("YAPIT_TRAINING_DISTRIBUTION", "").strip()
    if distribution:
        command += ["--distribution", distribution]
    command += ["--exec"]
    root, data, models, checkout = _wsl_paths(
        command,
        [
            str(workspace.root),
            str(workspace.data.root),
            str(models_directory()),
            str(upstream),
        ],
    )
    result = subprocess.run(
        [
            *command,
            "env",
            f"YAPIT_WORKSPACE={root}",
            f"YAPIT_DATA_DIR={data}",
            f"YAPIT_MODELS_DIR={models}",
            python,
            "-m",
            "yapit.cli",
            "voice-train",
            name,
            "--upstream",
            checkout,
            "--model",
            model,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=24 * 3600,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode:
        raise RuntimeError(
            "WSL training failed. Check the training status/log files and the configured "
            "Linux environment; the selected narration checkpoint is unchanged."
        )
    receipt = json.loads(result.stdout)
    if receipt.get("status") != "trained_needs_listening_review" or not receipt.get("checkpoints"):
        raise RuntimeError("Training returned no reviewable checkpoints.")
    fields = [key for key in ("job", "status_file", "transcripts") if receipt.get(key)]
    converted = _wsl_paths(
        command,
        [receipt[key] for key in fields] + receipt["checkpoints"],
        windows=True,
    )
    for key, path in zip(fields, converted[: len(fields)], strict=True):
        receipt[key] = path
    receipt["checkpoints"] = converted[len(fields) :]
    return receipt
