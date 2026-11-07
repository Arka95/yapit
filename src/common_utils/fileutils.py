from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path

from pydantic import BaseModel

from yapit.constants import DATA_FILENAME_TEMPLATE, DEFAULT_PROFILE


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


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


def data_filename(source: str, profile: str = DEFAULT_PROFILE, suffix: str = "") -> str:
    profile = safe_name(profile.strip().lower())
    stem = re.sub(r"[^a-z0-9._-]+", "_", source.lower()).strip("._-")[:140] or "narration"
    stem = stem.removeprefix(f"{profile}.") or "narration"
    suffix = suffix.lower()
    if suffix and not re.fullmatch(r"\.[a-z0-9][a-z0-9._-]*", suffix):
        raise ValueError("Invalid data filename extension.")
    return DATA_FILENAME_TEMPLATE.format(profile=profile, name=stem, suffix=suffix)


def reserve_artifact_name(
    source: str,
    profile: str,
    roots: list[Path],
    reservation: Path,
    suffix: str,
) -> str:
    base = data_filename(source, profile)
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


def copy_file_exclusive(source: Path, destination: Path) -> None:
    with destination.open("xb") as output:
        try:
            with source.open("rb") as incoming:
                shutil.copyfileobj(incoming, output)
        except OSError:
            output.close()
            destination.unlink(missing_ok=True)
            raise
