from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from yapit.common_utils.fileutils import (
    data_filename,
    file_sha256,
    save_json,
)
from yapit.common_utils.mediautils import SAMPLE_RATE, decode_audio, write_wav
from yapit.common_utils.metrics import timed
from yapit.constants import AUDIO_SAMPLE_NAME_TEMPLATE
from yapit.enrollment.artifacts import cleanup_session, session_directory
from yapit.enrollment.sample_scripts import ENROLLMENT_MAX_SECONDS, ENROLLMENT_MIN_SECONDS
from yapit.models import Sample, SampleManifest, Voice
from yapit.storage import DataStore

MIN_SAMPLES = 1


@contextmanager
def _reference_publication(
    prepared: Path, destination: Path, profile: Path, name: str
) -> Iterator[None]:
    """Replace the fixed reference and restore its inventories if publication fails.

    Stage the atomic audio swap on the destination filesystem.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        destination,
        *(
            profile / data_filename(kind, name, ".json")
            for kind in ("profile", "quality", "samples", "journal")
        ),
    ]
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        backups = {}
        for number, path in enumerate(paths):
            backup = staging / str(number) if path.exists() else None
            if backup is not None:
                shutil.copyfile(path, backup)
            backups[path] = backup
        incoming = staging / "incoming.wav"
        shutil.copyfile(prepared, incoming)
        try:
            incoming.replace(destination)
            yield
        except BaseException:
            for path, backup in backups.items():
                if backup is None:
                    path.unlink(missing_ok=True)
                else:
                    backup.replace(path)
            raise


def audio_quality(data: np.ndarray, *, min_seconds: float = 3, max_seconds: float = 20) -> dict:
    seconds = len(data) / SAMPLE_RATE
    peak = float(np.max(np.abs(data)))
    clipping = float(np.mean(np.abs(data) >= 0.999))
    window = SAMPLE_RATE // 50
    frames = data[: len(data) // window * window].reshape(-1, window)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    quiet = float(np.mean(rms < 0.005))
    overall_db = float(20 * np.log10(max(float(np.sqrt(np.mean(data**2))), 1e-10)))
    reasons = []
    if not min_seconds <= seconds <= max_seconds:
        reasons.append(f"Read the complete passage in {min_seconds:g}-{max_seconds:g} seconds.")
    if clipping > 0.001:
        reasons.append("More than 0.1% clipped samples; re-record without clipping.")
    if overall_db < -40:
        reasons.append("Signal is too quiet (RMS below -40 dBFS).")
    if quiet > 0.60:
        reasons.append("More than 60% very quiet frames; remove excessive silence.")
    return {
        "seconds": seconds,
        "peak": peak,
        "clipped_fraction": clipping,
        "quiet_fraction": quiet,
        "rms_dbfs": overall_db,
        "reasons": reasons,
    }


def profile_names(store: DataStore) -> list[str]:
    folders = store.root.iterdir()
    names = set()
    for path in folders:
        if not path.is_dir():
            continue
        try:
            store.profile_dir(path.name)
        except ValueError:
            continue
        if (
            any(
                (path / filename).exists()
                for filename in (
                    data_filename("profile", path.name, ".json"),
                    data_filename("samples", path.name, ".json"),
                )
            )
            or (
                session_directory(store, path.name) / data_filename("session", path.name, ".json")
            ).exists()
        ):
            names.add(path.name.lower())
    return sorted(names)


def save_journal(folder: Path, name: str, samples: list[Sample]) -> None:
    save_json(
        folder / data_filename("journal", name, ".json"),
        {
            "profile": name,
            "sample_count": len(samples),
            "minimum_samples": MIN_SAMPLES,
            "samples": [
                {
                    "audio": str(Path(sample.audio).resolve().relative_to(folder))
                    if Path(sample.audio).resolve().is_relative_to(folder)
                    else sample.audio,
                    "text": sample.text,
                    "sha256": file_sha256(Path(sample.audio)),
                }
                for sample in samples
            ],
        },
    )


@timed
def enroll(
    store: DataStore,
    name: str,
    manifest_path: Path,
    consent: bool = True,
    *,
    confined: bool = False,
) -> Voice:
    if not consent:
        raise ValueError("Explicit consent is required; enroll only your own authorized voice.")
    target = store.profile_dir(name)
    name = target.name
    existing = load_voice(store, name) if store.voice_path(name).exists() else None
    manifest_path = manifest_path.resolve()
    manifest = SampleManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if len(manifest.samples) != 1:
        raise ValueError(
            "Enrollment requires one complete reference recording. "
            "Use the Training tab for multi-clip datasets."
        )
    sample = manifest.samples[0]
    source = (manifest_path.parent / sample.audio).resolve()
    if confined and not source.is_relative_to(session_directory(store, name)):
        source = store.inside(source)
    if not source.is_file():
        raise ValueError(f"Missing sample: {source}")
    data = decode_audio(source, max_seconds=ENROLLMENT_MAX_SECONDS)
    quality = audio_quality(
        data, min_seconds=ENROLLMENT_MIN_SECONDS, max_seconds=ENROLLMENT_MAX_SECONDS
    )
    if quality["reasons"]:
        raise ValueError(f"{source.name}: {' '.join(quality['reasons'])}")
    audio_target = target / "audio"
    scratch = session_directory(store, name)
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".curate-", dir=scratch) as temp:
        staged = Path(temp)
        prepared = staged / "reference.wav"
        write_wav(prepared, data)
        digest = file_sha256(prepared)
        filename = data_filename(AUDIO_SAMPLE_NAME_TEMPLATE.format(number=1), name, ".wav")
        destination = audio_target / filename
        reference = Sample(audio=filename, text=sample.text)
        words = len(re.findall(r"\b[\w'-]+\b", sample.text))
        voice = (
            existing.model_copy(deep=True)
            if existing
            else Voice(
                name=name,
                consent=True,
                reference_audio=filename,
                reference_text=sample.text,
                samples=[reference],
                words_per_minute=min(210, max(100, words * 60 / quality["seconds"])),
            )
        )
        voice.name = name
        voice.samples = [reference]
        voice.reference_audio = filename
        voice.reference_text = sample.text
        voice.words_per_minute = min(210, max(100, words * 60 / quality["seconds"]))
        voice.checkpoint = voice.speaker = None
        if voice.__pydantic_extra__:
            voice.__pydantic_extra__.pop("pocket_voice_state", None)
        if existing and len(existing.samples) > 1:
            from yapit.training.dataset import preserve_legacy_samples

            preserve_legacy_samples(store, name, existing.samples)
        with _reference_publication(prepared, destination, target, name):
            save_json(target / data_filename("profile", name, ".json"), voice)
            save_json(
                target / data_filename("quality", name, ".json"),
                {
                    "samples": [{"audio": filename, "sha256": digest, **quality}],
                    "notice": "Signal checks only.",
                },
            )
            exported = [Sample(audio=str(destination), text=sample.text)]
            save_json(
                target / data_filename("samples", name, ".json"),
                {"samples": [sample.model_dump() for sample in exported]},
            )
            save_journal(target, name, exported)
    cleanup_session(store, name)
    return voice


def load_voice(store: DataStore, name: str) -> Voice:
    return Voice.model_validate_json(store.voice_path(name).read_text(encoding="utf-8"))
