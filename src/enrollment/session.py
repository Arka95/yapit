from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
from pydantic import Field, model_validator

from yapit.common_utils.fileutils import data_filename, file_sha256, save_json
from yapit.common_utils.mediautils import SAMPLE_RATE, decode_audio, write_wav
from yapit.common_utils.metrics import timed
from yapit.constants import AUDIO_SAMPLE_NAME_TEMPLATE
from yapit.enrollment.artifacts import session_directory
from yapit.enrollment.profiles import (
    audio_quality,
    enroll,
    load_voice,
)
from yapit.enrollment.sample_scripts import (
    ENROLLMENT_MAX_SECONDS,
    ENROLLMENT_MIN_SECONDS,
    Reading,
    load_readings,
)
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


class RecordingSession:
    """Resumable prompt/take pairing shared by enrollment and training collection."""

    training = False
    min_seconds = ENROLLMENT_MIN_SECONDS
    max_seconds = ENROLLMENT_MAX_SECONDS
    minimum_samples = 1

    def __init__(self, store: DataStore, alias: str):
        self.store = store
        self.folder = store.profile_dir(alias)
        self.alias = self.folder.name
        self.audio_folder = self.folder / "audio"
        self.scratch = session_directory(store, self.alias, training=self.training)
        self.path = self.scratch / data_filename("session", self.alias, ".json")
        self._selected: int | None = None
        if self.path.exists():
            self.draft = Draft.model_validate_json(self.path.read_text(encoding="utf-8"))
            if self.draft.alias.lower() != self.alias:
                raise ValueError("The saved enrollment alias does not match this folder.")
            self.draft.alias = self.alias
            for sample in filter(None, self.draft.samples):
                path = Path(sample.audio).resolve()
                if not path.is_relative_to(self.scratch):
                    path = store.inside(path)
                sample.audio = str(path)
        else:
            readings = self.load_readings()
            self.draft = Draft(
                alias=self.alias,
                readings=readings,
                samples=[None] * len(readings),
            )
            available = {sample.text: sample for sample in self.saved_samples()}
            for index, reading in enumerate(self.draft.readings):
                sample = available.get(reading.text)
                if sample is not None:
                    self.draft.samples[index] = sample
            self.draft.completed = self.finished
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
        return [sample for sample in self.draft.samples if sample]

    @property
    def ready(self) -> bool:
        return len(self.samples) >= self.minimum_samples

    def load_readings(self) -> list[Reading]:
        raise NotImplementedError

    def saved_samples(self) -> list[Sample]:
        raise NotImplementedError

    def select_sample(self, index: int) -> None:
        if not 0 <= index < len(self.draft.readings):
            raise ValueError("Choose a sample from the displayed passages.")
        self._selected = index

    def persist(self) -> None:
        if self.draft.completed:
            self.path.unlink(missing_ok=True)
        else:
            save_json(self.path, self.draft)

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

    @timed
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
        data = decode_audio(source, max_seconds=self.max_seconds)
        original_quality = audio_quality(
            data, min_seconds=self.min_seconds, max_seconds=self.max_seconds
        )
        if original_quality["seconds"] < self.min_seconds:
            raise ValueError(
                f"Record at least {self.min_seconds} seconds and read the complete passage."
            )
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
            if end - start >= self.min_seconds * SAMPLE_RATE:
                trim_start, trim_end = start / SAMPLE_RATE, (len(data) - end) / SAMPLE_RATE
                data = data[start:end]
            gain = 0.5 / max(float(np.max(np.abs(data))), 1e-10)
            if gain > 16:
                raise ValueError(
                    "The take is too quiet to normalize safely. Move closer and retry."
                )
            data = data * gain
            gain_db = float(20 * np.log10(gain))
        quality = audio_quality(data, min_seconds=self.min_seconds, max_seconds=self.max_seconds)
        if quality["reasons"]:
            raise ValueError(" ".join(quality["reasons"]))
        index = self.index
        prepared = self.scratch / data_filename(
            self.sample_name, self.alias, f".saved.{uuid.uuid4().hex}.wav"
        )
        temporary = self.scratch / data_filename(
            self.sample_name, self.alias, f".prepared.{uuid.uuid4().hex}.wav"
        )
        publication = prepared.with_name(f".{prepared.name}.{uuid.uuid4().hex}.tmp")
        try:
            write_wav(temporary, data)
            digest = file_sha256(temporary)
            if any(
                Path(sample.audio).resolve() != prepared.resolve()
                and file_sha256(Path(sample.audio)) == digest
                for position, sample in enumerate(self.draft.samples)
                if sample and position != index
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

    def completed(self) -> None:
        self.draft.completed = True
        self.persist()


class EnrollmentSession(RecordingSession):
    """Collect and publish exactly one paragraph as the profile's voice reference."""

    def load_readings(self) -> list[Reading]:
        return load_readings()

    def saved_samples(self) -> list[Sample]:
        if not self.store.voice_path(self.alias).exists():
            return []
        voice = load_voice(self.store, self.alias)
        if len(voice.samples) != 1:
            return []
        return [
            Sample(
                audio=str(self.store.inside(self.audio_folder / voice.reference_audio)),
                text=voice.reference_text,
            )
        ]

    @timed
    def finish(self) -> Voice:
        if not self.ready:
            raise ValueError("Save the complete enrollment paragraph before finishing.")
        manifest = self.scratch / data_filename("reference", self.alias, ".json")
        save_json(manifest, {"samples": [sample.model_dump() for sample in self.samples]})
        voice = enroll(self.store, self.alias, manifest, confined=True)
        self.draft.samples = self.saved_samples()
        self.completed()
        return voice
