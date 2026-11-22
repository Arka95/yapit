from __future__ import annotations

from pathlib import Path

from pydantic import Field

from yapit.models import Model, Text

ENROLLMENT_MIN_SECONDS = 45
ENROLLMENT_MAX_SECONDS = 90
ENROLLMENT_TARGET_SECONDS = 60


class Reading(Model):
    sample_id: str = ""
    title: str = Field(min_length=1, max_length=200)
    text: Text


def load_readings(directory: Path | None = None) -> list[Reading]:
    """Load the fixed enrollment paragraph and its exact reference transcript."""
    path = (directory or Path(__file__).parent) / "reference.json"
    return [Reading.model_validate_json(path.read_text(encoding="utf-8"))]
