from __future__ import annotations

import re
import unicodedata
from typing import Annotated

from pydantic import Field, model_validator

from yapit.models import Model

Percent = Annotated[float, Field(ge=0, le=100)]


def slug(value: str, fallback: str = "music") -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")[:48] or fallback


class BeatMusic(Model):
    beat: Annotated[int, Field(ge=0)]
    emotion: str
    energy: Annotated[float, Field(ge=0, le=1)]
    music_level: Percent
    reason: str


class MusicPlan(Model):
    emotion: str
    video_type: str
    bpm_min: Annotated[int, Field(ge=30, le=240)]
    bpm_max: Annotated[int, Field(ge=30, le=240)]
    instruments: list[str]
    queries: list[str]
    avoid: list[str]
    beats: list[BeatMusic]
    selection_note: str = "Metadata-based curation; listen before publishing."

    @model_validator(mode="after")
    def consistent_plan(self) -> MusicPlan:
        if self.bpm_min > self.bpm_max:
            raise ValueError("Minimum BPM must not exceed maximum BPM.")
        if len({beat.beat for beat in self.beats}) != len(self.beats):
            raise ValueError("Music plan beat IDs must be unique.")
        return self

    @property
    def beat_levels(self) -> dict[int, float]:
        return {beat.beat: beat.music_level for beat in self.beats}


class MusicCandidate(Model):
    id: str
    title: str
    creator: str
    source: str
    source_url: str
    audio_url: str
    license: str
    license_url: str
    duration: Annotated[float, Field(gt=0, le=600)] | None = None
    tags: list[str] = Field(default_factory=list)
    bpm: int | None = None
    score: float = 0
    reasons: list[str] = Field(default_factory=list)
