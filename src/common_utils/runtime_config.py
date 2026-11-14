"""Persist the local CPU/GPU choice without changing the caller's environment."""

import os
from pathlib import Path

from dotenv import dotenv_values, set_key

RUNTIME_MODES = ("cpu", "gpu")


def runtime_mode(root: Path) -> str:
    mode = os.environ.get("YAPIT_MODE")
    if mode is None:
        mode = dotenv_values(root / ".env").get("YAPIT_MODE") or "cpu"
    if mode not in RUNTIME_MODES:
        raise ValueError("YAPIT_MODE must be cpu or gpu; use yapit.cmd set-mode cpu or gpu.")
    return mode


def set_runtime_mode(root: Path, mode: str) -> dict:
    if mode not in RUNTIME_MODES:
        raise ValueError("Mode must be cpu or gpu.")
    config = root / ".env"
    set_key(config, "YAPIT_MODE", mode, quote_mode="never")
    return {
        "mode": mode,
        "config": str(config),
        "environment_override": os.environ.get("YAPIT_MODE") not in (None, mode),
    }
