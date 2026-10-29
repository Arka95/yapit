from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from pydantic import BaseModel


def save_json(path: Path, value: BaseModel | dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def safe_name(name: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name):
        raise ValueError("Name must contain 1-64 letters, digits, underscores or hyphens.")
    if name.upper().split(".")[0] in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise ValueError("Reserved Windows filename.")
    return name


class Workspace:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def inside(self, value: str | Path, *, exists: bool = True) -> Path:
        path = Path(value)
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("MCP paths must stay inside YAPIT_WORKSPACE.")
        if exists and not path.is_file():
            raise ValueError(f"File not found: {path}")
        return path

    def voice_path(self, name: str) -> Path:
        return self.inside(Path("voices") / safe_name(name) / "profile.json", exists=False)

    def job(self) -> Path:
        path = self.root / "outputs" / uuid.uuid4().hex
        path.mkdir(parents=True)
        return path
