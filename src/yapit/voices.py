from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from yapit.media import SAMPLE_RATE, decode_audio, write_wav
from yapit.models import Sample, SampleManifest, Voice
from yapit.storage import Workspace, save_json


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


def enroll(
    workspace: Workspace,
    name: str,
    manifest_path: Path,
    consent: bool,
    *,
    confined: bool = False,
) -> Voice:
    if not consent:
        raise ValueError("Explicit consent is required; enroll only your own authorized voice.")
    target = workspace.voice_path(name).parent
    if target.exists():
        raise ValueError(f"Voice {name!r} exists; use a new name to preserve recordings.")
    manifest_path = manifest_path.resolve()
    manifest = SampleManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="curate-", dir=target.parent) as temp:
        staged = Path(temp)
        samples = []
        report = []
        seen = set()
        for index, sample in enumerate(manifest.samples):
            source = (manifest_path.parent / sample.audio).resolve()
            if confined:
                source = workspace.inside(source)
            if not source.is_file():
                raise ValueError(f"Missing sample: {source}")
            data = decode_audio(source, max_seconds=20)
            quality = audio_quality(data)
            if quality["reasons"]:
                raise ValueError(f"{source.name}: {' '.join(quality['reasons'])}")
            digest = hashlib.sha256(data.tobytes()).hexdigest()
            if digest in seen:
                raise ValueError(f"Duplicate recording: {source.name}")
            seen.add(digest)
            relative = f"sample-{index:04}.wav"
            write_wav(staged / relative, data)
            samples.append(Sample(audio=relative, text=sample.text))
            report.append({"audio": relative, "sha256": digest, **quality})
        # This is a signal-level ranking, not a guarantee of speech/noise/accent quality.
        best = min(
            range(len(report)),
            key=lambda i: report[i]["quiet_fraction"] + abs(report[i]["seconds"] - 10) / 40,
        )
        words = sum(len(re.findall(r"\b[\w'-]+\b", sample.text)) for sample in samples)
        seconds = sum(item["seconds"] for item in report)
        wpm = min(210, max(100, words * 60 / seconds))
        voice = Voice(
            name=name,
            consent=True,
            reference_audio=samples[best].audio,
            reference_text=samples[best].text,
            samples=samples,
            words_per_minute=wpm,
        )
        save_json(staged / "profile.json", voice)
        save_json(
            staged / "quality.json",
            {
                "samples": report,
                "notice": (
                    "Signal checks only. Review music, noise, speakers and transcripts by ear."
                ),
            },
        )
        shutil.copytree(staged, target)
    return voice


def load_voice(workspace: Workspace, name: str) -> Voice:
    return Voice.model_validate_json(workspace.voice_path(name).read_text(encoding="utf-8"))
