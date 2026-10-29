from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Text = Annotated[str, Field(min_length=1, max_length=4000)]
Seconds = Annotated[float, Field(ge=5, le=180, allow_inf_nan=False)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Brief(Model):
    topic: Annotated[str, Field(max_length=8000)] = ""
    kind: Literal["travel", "product"] = "travel"
    seconds: Seconds = 30
    facts: Annotated[str, Field(max_length=16000)] = ""
    language: Literal["English"] = "English"
    words_per_minute: Annotated[float, Field(ge=80, le=240)] = 150


class Beat(Model):
    text: Text
    visual: Annotated[str, Field(min_length=1, max_length=500)] = "Supporting B-roll"
    pause_ms: Annotated[int, Field(ge=0, le=1200)] = 180


class Script(Model):
    title: Annotated[str, Field(min_length=1, max_length=200)]
    beats: Annotated[list[Beat], Field(min_length=1, max_length=40)]

    @model_validator(mode="after")
    def bounded_text(self) -> Script:
        if self.word_count > 650:
            raise ValueError("Script exceeds 650 words; split it into multiple reels.")
        return self

    @property
    def word_count(self) -> int:
        return sum(len(re.findall(r"\b[\w'-]+\b", beat.text)) for beat in self.beats)

    def estimate_seconds(self, words_per_minute: float) -> float:
        return self.word_count * 60 / words_per_minute + sum(
            beat.pause_ms / 1000 for beat in self.beats
        )

    @classmethod
    def from_text(cls, text: str) -> Script:
        parts = re.split(r"\n\s*\n|(?<=[.!?])\s+", text.strip())
        return cls(title="Your commentary", beats=[Beat(text=p) for p in parts if p.strip()])


class Sample(Model):
    audio: str
    text: Text


class SampleManifest(Model):
    samples: Annotated[list[Sample], Field(min_length=1, max_length=300)]


class Voice(Model):
    name: str
    consent: Literal[True]
    reference_audio: str
    reference_text: str
    samples: list[Sample]
    words_per_minute: float
    checkpoint: str | None = None
    speaker: str | None = None


class Cue(Model):
    start: float
    end: float
    text: str
    visual: str
    beat: int


class Timeline(Model):
    seconds: Annotated[float, Field(gt=0, le=190)]
    width: Annotated[int, Field(ge=64, le=2160)] = 1080
    height: Annotated[int, Field(ge=64, le=3840)] = 1920
    fps: Annotated[int, Field(ge=1, le=60)] = 30
    timing_method: str = (
        "measured beats and pauses; quiet-window sub-cuts; approximate phrase captions"
    )
    cues: Annotated[list[Cue], Field(min_length=1, max_length=200)]

    @model_validator(mode="after")
    def continuous(self) -> Timeline:
        if not self.cues or self.seconds <= 0:
            raise ValueError("Timeline must contain positive-duration cues.")
        previous = 0.0
        for cue in self.cues:
            if abs(cue.start - previous) > 0.001 or cue.end <= cue.start:
                raise ValueError("Timeline must be contiguous and ordered.")
            previous = cue.end
        if abs(previous - self.seconds) > 0.001:
            raise ValueError("Timeline must cover the entire audio.")
        return self
