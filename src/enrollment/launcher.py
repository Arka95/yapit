from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from yapit.storage import Workspace


def enroll_voice(*, workspace: Workspace | None = None) -> dict:
    """Launch the recording wizard independently without inheriting MCP streams."""
    if sys.platform != "win32":
        raise RuntimeError("The enrollment app requires an interactive Windows desktop.")
    workspace = workspace or Workspace(os.environ.get("YAPIT_WORKSPACE", "workspace"))
    executable = Path(sys.executable)
    windowed = executable.with_name("pythonw.exe")
    if windowed.is_file():
        executable = windowed
    process = subprocess.Popen(
        [str(executable), "-m", "yapit.enrollment.desktop"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={
            **os.environ,
            "YAPIT_WORKSPACE": str(workspace.root),
            "YAPIT_DATA_DIR": str(workspace.data.root),
        },
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    return {
        "status": "launched",
        "pid": process.pid,
        "next": "Choose or create a profile, record and review takes, then finish in the app.",
    }
