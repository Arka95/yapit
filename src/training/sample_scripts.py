"""Short, numbered passages for collecting a fine-tuning dataset."""

from __future__ import annotations

import re
from pathlib import Path

from yapit.enrollment.sample_scripts import Reading


def load_readings(directory: Path | None = None) -> list[Reading]:
    """Load training prompts in numeric order, preserving passage identifiers."""
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
        raise ValueError("Provide between 5 and 100 training sample scripts.")
    return [reading for _, reading in sorted(readings)]
