from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
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


def reserve_artifact_name(
    source: str,
    profile: str,
    roots: list[Path],
    reservation: Path,
    suffix: str,
) -> str:
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", source).strip("._-")[:140] or "narration"
    profile = safe_name(profile)
    stem = stem.removeprefix(f"{profile}.") or "narration"
    base = f"{profile}.{stem}"
    reservation.mkdir(parents=True, exist_ok=True)
    number = 0
    while True:
        name = f"{base}.{number}" if number else base
        if any(list(root.glob(f"{name}.*")) for root in roots):
            number += 1
            continue
        try:
            with (reservation / f"{name}.{suffix}").open("x", encoding="utf-8") as claim:
                json.dump({"status": "reserved", "name": name}, claim)
            return name
        except FileExistsError:
            number += 1


class OutputArtifacts:
    def __init__(self, workspace: Workspace, source: str, profile: str):
        self.metadata = workspace.root / "outputs"
        self.audio = workspace.data.generated_audio
        self.transcripts = workspace.data.transcripts
        self.name = reserve_artifact_name(
            source,
            profile,
            [self.metadata, self.audio, self.transcripts],
            self.metadata,
            "status.json",
        )

    def path(self, suffix: str, directory: Path | None = None) -> Path:
        return (directory or self.metadata) / f"{self.name}.{suffix}"


class Workspace:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.data = DataStore(self.root.parent / "data")

    def inside(self, value: str | Path, *, exists: bool = True) -> Path:
        path = Path(value)
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(self.root) and not path.is_relative_to(self.data.root):
            raise ValueError("MCP paths must stay inside YAPIT_WORKSPACE or YAPIT_DATA_DIR.")
        if exists and not path.is_file():
            raise ValueError(f"File not found: {path}")
        return path

    def voice_path(self, name: str) -> Path:
        return self.inside(Path("voices") / safe_name(name) / "profile.json", exists=False)

    def output(self, source: str, profile: str = "arka") -> OutputArtifacts:
        return OutputArtifacts(self, source, profile)

    def voice_audio_dir(self, name: str) -> Path:
        directory = self.data.audio / "voices" / safe_name(name)
        return directory if directory.exists() else self.voice_path(name).parent


class DataStore:
    def __init__(self, default: Path | None = None):
        fallback = (
            default
            or Path(os.environ.get("YAPIT_WORKSPACE", "workspace")).resolve().parent / "data"
        )
        self.root = Path(os.environ.get("YAPIT_DATA_DIR") or fallback).resolve()
        self.audio = self.root / "audio"
        self.transcripts = self.root / "transcripts"
        self.generated_audio = self.root / "generated_audio"
        for directory in (self.audio, self.transcripts, self.generated_audio):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def within(directory: Path, output: Path | None = None) -> Path:
        path = (output if output is not None else directory).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError(f"Output must stay inside {directory}.")
        return path

    def archive_recording(self, source: Path) -> Path:
        source = source.resolve()
        if source.is_relative_to(self.audio) or source.is_relative_to(self.generated_audio):
            return source
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        target = self.audio / "imports" / digest / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(
                    f"Archived recording was modified; refusing to overwrite {target}."
                )
            return target
        try:
            output = target.open("xb")
        except FileExistsError:
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Recording archive collision: {target}.") from None
            return target
        with output:
            try:
                with source.open("rb") as incoming:
                    shutil.copyfileobj(incoming, output)
            except OSError:
                output.close()
                target.unlink(missing_ok=True)
                raise
        return target
