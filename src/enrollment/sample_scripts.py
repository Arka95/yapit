from __future__ import annotations

import re
from pathlib import Path

from pydantic import Field

from yapit.models import Model, Text


class Reading(Model):
    sample_id: str = ""
    title: str = Field(min_length=1, max_length=200)
    text: Text


def load_readings(directory: Path | None = None) -> list[Reading]:
    directory = directory or Path(__file__).parent / "sampleScripts"
    readings = []
    for path in directory.glob("*.json"):
        match = re.fullmatch(r"sample_([1-9][0-9]*)", path.stem)
        if not match:
            raise ValueError(f"Invalid sample script filename: {path.name}")
        reading = Reading.model_validate_json(path.read_text(encoding="utf-8"))
        if reading.sample_id and reading.sample_id != path.stem:
            raise ValueError(f"Sample ID does not match filename: {path.name}")
        reading.sample_id = path.stem
        readings.append((int(match[1]), reading))
    if not 5 <= len(readings) <= 100:
        raise ValueError("Provide between 5 and 100 static sample scripts.")
    return [reading for _, reading in sorted(readings)]
