from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from yapit.common_utils.fileutils import data_filename, file_sha256, save_json
from yapit.common_utils.mediautils import SAMPLE_RATE, decode_audio, write_wav
from yapit.enrollment.artifacts import cleanup_session, session_directory
from yapit.models import Sample, SampleManifest, Voice
from yapit.storage import DataStore

MIN_SAMPLES = 5


def audio_quality(data: np.ndarray) -> dict:
    seconds = len(data) / SAMPLE_RATE
    peak = float(np.max(np.abs(data)))
    clipping = float(np.mean(np.abs(data) >= 0.999))
    window = SAMPLE_RATE // 50
    frames = data[: len(data) // window * window].reshape(-1, window)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    quiet = float(np.mean(rms < 0.005))
    overall_db = float(20 * np.log10(max(float(np.sqrt(np.mean(data**2))), 1e-10)))
    reasons = []
    if not 3 <= seconds <= 20:
        reasons.append("Use complete, manually segmented clips between 3 and 20 seconds.")
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
        if any(
            (path / filename).exists()
            for filename in (
                data_filename("profile", path.name, ".json"),
                data_filename("samples", path.name, ".json"),
            )
        ):
            names.add(path.name.lower())
    return sorted(names)


def audio_name(name: str, source: Path) -> str:
    return data_filename(source.stem, name, ".wav")


def merged_samples(store: DataStore, name: str, additions: list[Sample]) -> list[Sample]:
    samples = {}
    if store.voice_path(name).exists():
        voice = load_voice(store, name)
        directory = store.voice_audio_dir(name)
        for sample in voice.samples:
            samples[audio_name(name, Path(sample.audio))] = Sample(
                audio=str(store.inside(directory / sample.audio)), text=sample.text
            )
    for sample in additions:
        samples[audio_name(name, Path(sample.audio))] = sample
    return list(samples.values())


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
    additions = {}
    for sample in manifest.samples:
        source = (manifest_path.parent / sample.audio).resolve()
        if confined:
            source = store.inside(source)
        if not source.is_file():
            raise ValueError(f"Missing sample: {source}")
        filename = audio_name(name, source)
        additions[filename] = Sample(audio=str(source), text=sample.text)
    combined = merged_samples(store, name, list(additions.values()))
    audio_target = target / "audio"
    scratch = session_directory(store, name)
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".curate-", dir=scratch) as temp:
        staged = Path(temp)
        samples, report, seen = [], [], set()
        for sample in combined:
            source = Path(sample.audio)
            data = decode_audio(source, max_seconds=20)
            quality = audio_quality(data)
            if quality["reasons"]:
                raise ValueError(f"{source.name}: {' '.join(quality['reasons'])}")
            digest = hashlib.sha256(data.tobytes()).hexdigest()
            if digest in seen:
                raise ValueError(f"Duplicate recording: {source.name}")
            seen.add(digest)
            filename = audio_name(name, source)
            write_wav(staged / filename, data)
            samples.append(Sample(audio=filename, text=sample.text))
            report.append({"audio": filename, "sha256": digest, **quality})
        if len(samples) < MIN_SAMPLES:
            raise ValueError(f"Save at least {MIN_SAMPLES} distinct samples before enrollment.")
        best = min(
            range(len(report)),
            key=lambda i: report[i]["quiet_fraction"] + abs(report[i]["seconds"] - 10) / 40,
        )
        words = sum(len(re.findall(r"\b[\w'-]+\b", sample.text)) for sample in samples)
        seconds = sum(item["seconds"] for item in report)
        voice = (
            existing.model_copy(deep=True)
            if existing
            else Voice(
                name=name,
                consent=True,
                reference_audio=samples[best].audio,
                reference_text=samples[best].text,
                samples=samples,
                words_per_minute=min(210, max(100, words * 60 / seconds)),
            )
        )
        voice.name = name
        voice.samples = samples
        voice.reference_audio = samples[best].audio
        voice.reference_text = samples[best].text
        voice.words_per_minute = min(210, max(100, words * 60 / seconds))
        audio_target.mkdir(parents=True, exist_ok=True)
        for path in staged.iterdir():
            shutil.copyfile(path, audio_target / path.name)
        save_json(target / data_filename("profile", name, ".json"), voice)
        save_json(
            target / data_filename("quality", name, ".json"),
            {"samples": report, "notice": "Signal checks only."},
        )
        exported = [
            Sample(audio=str(audio_target / sample.audio), text=sample.text) for sample in samples
        ]
        save_json(
            target / data_filename("samples", name, ".json"),
            {"samples": [sample.model_dump() for sample in exported]},
        )
        save_journal(target, name, exported)
    cleanup_session(store, name)
    return voice


def load_voice(store: DataStore, name: str) -> Voice:
    return Voice.model_validate_json(store.voice_path(name).read_text(encoding="utf-8"))
