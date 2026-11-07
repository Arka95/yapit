from __future__ import annotations

import os
from pathlib import Path

from yapit.storage import DataStore


def session_directory(store: DataStore, alias: str) -> Path:
    name = store.profile_dir(alias).name
    workspace = Path(os.environ.get("YAPIT_WORKSPACE") or store.root.parent / "workspace").resolve()
    directory = (workspace / "enrollment" / name).resolve()
    if not directory.is_relative_to(workspace) or directory.is_relative_to(store.root):
        raise ValueError("Enrollment workspace must stay outside YAPIT_DATA_DIR.")
    return directory


def cleanup_session(store: DataStore, alias: str) -> None:
    directory = session_directory(store, alias)
    if not directory.exists():
        return
    for path in directory.iterdir():
        if path.is_file() and path.name.startswith(f"{directory.name}."):
            path.unlink()
    if not any(directory.iterdir()):
        directory.rmdir()
