"""Training-only recordings, kept separate from the profile's cloning reference."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from yapit.common_utils.fileutils import copy_file_exclusive, data_filename, file_sha256, save_json
from yapit.common_utils.mediautils import decode_audio, write_wav
from yapit.enrollment.profiles import audio_quality, load_voice, save_journal
from yapit.models import Sample, SampleManifest
from yapit.storage import DataStore

MIN_TRAINING_SAMPLES = 10
MIN_TRAINING_SECONDS = 60
TRAINING_MAX_SECONDS = 20


def training_manifest(store: DataStore, name: str) -> Path:
    folder = store.profile_dir(name)
    return folder / "training" / data_filename("samples", folder.name, ".json")


def load_training_samples(store: DataStore, name: str) -> list[Sample]:
    """Resolve a saved dataset, falling back only to legacy multi-clip profiles."""
    path = training_manifest(store, name)
    if path.exists():
        samples = SampleManifest.model_validate_json(path.read_text(encoding="utf-8")).samples
        directory = path.parent
    else:
        voice = load_voice(store, name)
        samples = voice.samples if len(voice.samples) > 1 else []
        directory = store.voice_audio_dir(name)
    return [
        Sample(
            audio=str(store.inside(directory / sample.audio.replace("\\", os.sep))),
            text=sample.text,
        )
        for sample in samples
    ]


def preserve_legacy_samples(store: DataStore, name: str, samples: list[Sample]) -> list[Sample]:
    """Retain the legacy collection before replacing a profile with one reference."""
    path = training_manifest(store, name)
    saved = load_training_samples(store, name) if path.exists() else []
    directory = store.voice_audio_dir(name)
    target = path.parent / "audio"
    target.mkdir(parents=True, exist_ok=True)
    combined = {}
    for sample in samples:
        source = store.inside(directory / sample.audio)
        destination = target / source.name
        if destination.exists():
            if file_sha256(source) != file_sha256(destination):
                raise ValueError("Training archive collision; existing recordings preserved.")
        else:
            copy_file_exclusive(source, destination)
        combined[str(destination)] = sample.text
    combined.update({sample.audio: sample.text for sample in saved})
    save_json(
        path,
        SampleManifest(
            samples=[
                Sample(audio=str(Path(audio).relative_to(path.parent)), text=text)
                for audio, text in combined.items()
            ]
        ),
    )
    return load_training_samples(store, name)


def migrate_profile_recordings(store: DataStore, name: str) -> dict:
    """Separate a legacy dataset, retaining its current reference and metadata."""
    voice = load_voice(store, name)
    folder = store.profile_dir(name)
    if len(voice.samples) <= 1:
        return {"profile": voice.name, "status": "already_separated"}
    samples = list(voice.samples)
    audio = store.voice_audio_dir(name)
    protected = {store.inside(audio / voice.reference_audio)}
    protected.update(store.inside(reference.audio) for reference in voice.styles.values())
    archive = training_manifest(store, name).parent
    archive.mkdir(parents=True, exist_ok=True)
    for kind in ("profile", "samples", "quality", "journal"):
        source = folder / data_filename(kind, voice.name, ".json")
        destination = archive / data_filename(f"legacy.{kind}", voice.name, ".json")
        if source.exists() and not destination.exists():
            copy_file_exclusive(source, destination)
    training = preserve_legacy_samples(store, name, samples)
    voice.samples = [Sample(audio=voice.reference_audio, text=voice.reference_text)]
    save_json(store.voice_path(name), voice)
    reference = Sample(audio=str(audio / voice.reference_audio), text=voice.reference_text)
    save_json(
        folder / data_filename("samples", voice.name, ".json"),
        SampleManifest(samples=[reference]),
    )
    save_journal(folder, voice.name, [reference])
    quality = folder / data_filename("quality", voice.name, ".json")
    if quality.exists():
        payload = json.loads(quality.read_text(encoding="utf-8"))
        payload["samples"] = [
            row for row in payload.get("samples", []) if row.get("audio") == voice.reference_audio
        ]
        save_json(quality, payload)
    moved = 0
    for sample in samples:
        source = store.inside(audio / sample.audio)
        if source in protected:
            continue
        destination = archive / "audio" / source.name
        if file_sha256(source) != file_sha256(destination):
            raise ValueError("Training archive verification failed; original retained.")
        source.unlink()
        moved += 1
    return {
        "profile": voice.name,
        "status": "separated",
        "training_clips": len(training),
        "moved_files": moved,
        "reference": voice.reference_audio,
        "new_enrollment_recording_required": True,
    }


def save_training_samples(store: DataStore, name: str, samples: list[Sample]) -> list[Sample]:
    """Validate and publish a distinct short-clip dataset without changing the voice."""
    if len(samples) < MIN_TRAINING_SAMPLES:
        raise ValueError(f"Save at least {MIN_TRAINING_SAMPLES} distinct training clips.")
    folder = store.profile_dir(name)
    manifest = training_manifest(store, name)
    audio_folder = manifest.parent / "audio"
    audio_folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".training-", dir=folder) as temporary:
        staged = Path(temporary)
        prepared, seen, seconds = [], set(), 0.0
        for sample in samples:
            data = decode_audio(Path(sample.audio), max_seconds=TRAINING_MAX_SECONDS)
            quality = audio_quality(data)
            if quality["reasons"]:
                raise ValueError(f"{Path(sample.audio).name}: {' '.join(quality['reasons'])}")
            digest = hashlib.sha256(data.tobytes()).hexdigest()
            if digest in seen:
                raise ValueError("Duplicate training recording.")
            seen.add(digest)
            seconds += quality["seconds"]
            filename = data_filename(f"training.{digest[:16]}", folder.name, ".wav")
            write_wav(staged / filename, data)
            prepared.append(Sample(audio=str(audio_folder / filename), text=sample.text))
        if seconds < MIN_TRAINING_SECONDS:
            raise ValueError(f"Training needs at least {MIN_TRAINING_SECONDS} seconds of clips.")
        for sample in prepared:
            source, destination = staged / Path(sample.audio).name, Path(sample.audio)
            if destination.exists():
                if file_sha256(source) != file_sha256(destination):
                    raise ValueError("Training audio collision; existing recordings preserved.")
            else:
                copy_file_exclusive(source, destination)
        save_json(
            manifest,
            SampleManifest(
                samples=[
                    Sample(
                        audio=str(Path(sample.audio).relative_to(manifest.parent)),
                        text=sample.text,
                    )
                    for sample in prepared
                ]
            ),
        )
    return prepared
