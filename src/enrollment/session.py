from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path

import numpy as np
from pydantic import Field, model_validator

from yapit.common_utils.fileutils import data_filename, save_json
from yapit.common_utils.mediautils import SAMPLE_RATE, decode_audio, write_wav
from yapit.constants import AUDIO_SAMPLE_NAME_TEMPLATE
from yapit.enrollment.artifacts import session_directory
from yapit.enrollment.profiles import (
    MIN_SAMPLES,
    audio_name,
    audio_quality,
    enroll,
    load_voice,
    merged_samples,
    save_journal,
)
from yapit.enrollment.sample_scripts import Reading, load_readings
from yapit.models import Model, Sample, Voice
from yapit.storage import DataStore


class Draft(Model):
    alias: str
    readings: list[Reading] = Field(min_length=1, max_length=100)
    samples: list[Sample | None]
    completed: bool = False

    @model_validator(mode="after")
    def matching_samples(self) -> Draft:
        if len(self.readings) != len(self.samples):
            raise ValueError("Enrollment draft has inconsistent sample slots.")
        return self


class EnrollmentSession:
    def __init__(self, store: DataStore, alias: str):
        self.store = store
        self.folder = store.profile_dir(alias)
        self.alias = self.folder.name
        self.audio_folder = self.folder / "audio"
        self.scratch = session_directory(store, self.alias)
        self.path = self.scratch / data_filename("session", self.alias, ".json")
        self._selected: int | None = None
        if self.path.exists():
            self.draft = Draft.model_validate_json(self.path.read_text(encoding="utf-8"))
            if self.draft.alias.lower() != self.alias:
                raise ValueError("The saved enrollment alias does not match this folder.")
            self.draft.alias = self.alias
            for sample in self.draft.samples:
                if sample:
                    sample.audio = str(store.inside(self.folder / sample.audio))
        else:
            readings = load_readings()
            self.draft = Draft(
                alias=self.alias,
                readings=readings,
                samples=[None] * len(readings),
                completed=store.voice_path(self.alias).exists(),
            )
        if store.voice_path(self.alias).exists():
            voice = load_voice(store, self.alias)
            directory = store.voice_audio_dir(self.alias)
            available = {
                audio_name(self.alias, Path(sample.audio)): sample for sample in voice.samples
            }
            for index, reading in enumerate(self.draft.readings):
                number = (
                    reading.sample_id.removeprefix("sample_") if reading.sample_id else index + 1
                )
                filename = data_filename(
                    AUDIO_SAMPLE_NAME_TEMPLATE.format(number=number), self.alias, ".wav"
                )
                sample = available.get(filename)
                if sample is not None:
                    self.draft.samples[index] = (
                        Sample(audio=str(directory / sample.audio), text=sample.text)
                        if sample.text == reading.text
                        else None
                    )
        self.folder.mkdir(parents=True, exist_ok=True)
        self.persist()

    @property
    def index(self) -> int:
        if self._selected is not None:
            return self._selected
        return next(
            (index for index, sample in enumerate(self.draft.samples) if sample is None),
            len(self.draft.samples),
        )

    @property
    def finished(self) -> bool:
        return all(sample is not None for sample in self.draft.samples)

    @property
    def samples(self) -> list[Sample]:
        return merged_samples(
            self.store, self.alias, [sample for sample in self.draft.samples if sample]
        )

    @property
    def ready(self) -> bool:
        return len(self.samples) >= MIN_SAMPLES

    def select_sample(self, index: int) -> None:
        if not 0 <= index < len(self.draft.readings):
            raise ValueError("Choose a sample from the displayed passages.")
        self._selected = index

    def persist(self) -> None:
        if self.draft.completed:
            self.path.unlink(missing_ok=True)
        else:
            save_json(self.path, self.draft)
        samples = self.samples
        save_json(
            self.folder / data_filename("samples", self.alias, ".json"),
            {"samples": [s.model_dump() for s in samples]},
        )
        save_journal(self.folder, self.alias, samples)

    def new_take(self) -> Path:
        if self.index == len(self.draft.readings):
            raise ValueError("Choose a passage to record another sample.")
        if self.draft.completed:
            self.draft.completed = False
            self.persist()
        return self.scratch / data_filename(
            self.sample_name,
            self.alias,
            f".raw.{uuid.uuid4().hex[:12]}.wav",
        )

    @property
    def sample_name(self) -> str:
        reading = self.draft.readings[self.index]
        number = reading.sample_id.removeprefix("sample_") if reading.sample_id else self.index + 1
        return AUDIO_SAMPLE_NAME_TEMPLATE.format(number=number)

    def save_take(self, source: Path, confirmed: bool, normalize: bool) -> dict:
        if not confirmed:
            raise ValueError(
                "Replay the take and confirm that you read the displayed text exactly."
            )
        if self.index == len(self.draft.readings):
            raise ValueError("Choose a passage to record another sample.")
        source = source.resolve()
        if not source.is_relative_to(self.scratch):
            raise ValueError("The recording must stay inside this enrollment session workspace.")
        data = decode_audio(source, max_seconds=20)
        original_quality = audio_quality(data)
        if original_quality["seconds"] < 3:
            raise ValueError("Record at least 3 seconds and read the complete passage.")
        if original_quality["clipped_fraction"] > 0.001:
            raise ValueError("This take is clipped. Reduce microphone gain and record again.")
        gain_db = trim_start = trim_end = 0.0
        if normalize:
            window = SAMPLE_RATE // 50
            frames = data[: len(data) // window * window].reshape(-1, window)
            active = np.sqrt(np.mean(frames**2, axis=1)) > 0.002
            sustained = (
                np.convolve(active.astype(np.int32), np.ones(10, dtype=np.int32), mode="same") >= 4
            )
            indices = np.flatnonzero(sustained)
            if not len(indices):
                raise ValueError("No sustained sound detected. Check the microphone and try again.")
            start = max(0, int(indices[0]) * window - SAMPLE_RATE // 2)
            end = min(len(data), (int(indices[-1]) + 1) * window + SAMPLE_RATE // 2)
            if end - start >= 3 * SAMPLE_RATE:
                trim_start, trim_end = start / SAMPLE_RATE, (len(data) - end) / SAMPLE_RATE
                data = data[start:end]
            gain = 0.5 / max(float(np.max(np.abs(data))), 1e-10)
            if gain > 16:
                raise ValueError(
                    "The take is too quiet to normalize safely. Move closer and retry."
                )
            data = data * gain
            gain_db = float(20 * np.log10(gain))
        quality = audio_quality(data)
        if quality["reasons"]:
            raise ValueError(" ".join(quality["reasons"]))
        index = self.index
        self.audio_folder.mkdir(parents=True, exist_ok=True)
        prepared = self.audio_folder / data_filename(self.sample_name, self.alias, ".wav")
        temporary = self.scratch / data_filename(
            self.sample_name, self.alias, f".prepared.{uuid.uuid4().hex}.wav"
        )
        publication = prepared.with_name(f".{prepared.name}.{uuid.uuid4().hex}.tmp")
        try:
            write_wav(temporary, data)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if any(
                Path(sample.audio).resolve() != prepared.resolve()
                and hashlib.sha256(Path(sample.audio).read_bytes()).hexdigest() == digest
                for sample in self.samples
            ):
                raise ValueError("This duplicates a saved recording. Read the current passage.")
            shutil.copyfile(temporary, publication)
            publication.replace(prepared)
        finally:
            temporary.unlink(missing_ok=True)
            publication.unlink(missing_ok=True)
        self.draft.samples[index] = Sample(
            audio=str(prepared), text=self.draft.readings[index].text
        )
        self.draft.completed = False
        report = {
            "raw": source.name,
            "prepared": prepared.name,
            "quality": quality,
            "gain_db": gain_db,
            "trim_start_seconds": trim_start,
            "trim_end_seconds": trim_end,
            "transcript_confirmed": True,
            "notice": "Signal checks only. Listen for pronunciation, speaker identity and noise.",
        }
        save_json(self.scratch / f"{prepared.stem}.quality.json", report)
        self.persist()
        self._selected = None
        return report

    def finish(self) -> Voice:
        if not self.ready:
            raise ValueError(f"Save at least {MIN_SAMPLES} distinct samples before enrollment.")
        self.persist()
        voice = enroll(
            self.store,
            self.alias,
            self.folder / data_filename("samples", self.alias, ".json"),
            confined=True,
        )
        self.draft.completed = True
        self.persist()
        return voice
