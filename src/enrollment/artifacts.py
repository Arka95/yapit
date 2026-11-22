from __future__ import annotations

import os
from pathlib import Path

from yapit.storage import DataStore


def session_directory(store: DataStore, alias: str, *, training: bool = False) -> Path:
    name = store.profile_dir(alias).name
    workspace = Path(os.environ.get("YAPIT_WORKSPACE") or store.root.parent / "workspace").resolve()
    directory = (
        workspace / "training" / name if training else workspace / "enrollment" / name / "reference"
    ).resolve()
    if not directory.is_relative_to(workspace) or directory.is_relative_to(store.root):
        raise ValueError("Enrollment workspace must stay outside YAPIT_DATA_DIR.")
    return directory


def cleanup_session(store: DataStore, alias: str, *, training: bool = False) -> None:
    directory = session_directory(store, alias, training=training)
    if not directory.exists():
        return
    for path in directory.iterdir():
        prefixes = (f"{store.profile_dir(alias).name}.", f"{directory.name}.microphone.")
        if path.is_file() and path.name.startswith(prefixes):
            path.unlink()
    if not any(directory.iterdir()):
        directory.rmdir()
