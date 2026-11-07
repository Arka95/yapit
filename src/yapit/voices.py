from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from yapit.media import SAMPLE_RATE, decode_audio, write_wav
from yapit.models import DeliveryStyle, Sample, SampleManifest, StyleReference, Voice
from yapit.storage import Workspace, safe_name, save_json


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
    audio_target = workspace.data.audio / "voices" / safe_name(name)
    if audio_target.exists():
        raise ValueError(f"Voice audio {name!r} exists; choose a new alias.")
    audio_target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="curate-", dir=audio_target.parent) as temp:
        staged = Path(temp)
        (staged / "raw").mkdir()
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
            shutil.copyfile(source, staged / "raw" / f"sample-{index:04}{source.suffix}")
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
        shutil.copytree(staged, audio_target)
        save_json(target / "profile.json", voice)
        save_json(
            target / "quality.json",
            {
                "samples": report,
                "notice": (
                    "Signal checks only. Review music, noise, speakers and transcripts by ear."
                ),
            },
        )
        save_json(
            workspace.data.transcripts / "voices" / name / "samples.json",
            {
                "samples": [
                    {"audio": str(audio_target / sample.audio), "text": sample.text}
                    for sample in samples
                ]
            },
        )
    return voice


def load_voice(workspace: Workspace, name: str) -> Voice:
    return Voice.model_validate_json(workspace.voice_path(name).read_text(encoding="utf-8"))


STYLE_GUIDES = {
    DeliveryStyle.lively_vlog: {
        "direction": (
            "Sound warmly excited and curious, as if showing a friend something delightful. "
            "Smile naturally, vary emphasis and pitch, and keep the wildlife advice firm but "
            "friendly. No shouting or exaggerated announcer voice."
        ),
        "text": (
            "Look at these little faces! They're curious, fluffy, and full of character. "
            "Let's enjoy the moment, keep a respectful distance, and let them carry on "
            "with their day."
        ),
    },
    DeliveryStyle.relaxed_travel: {
        "direction": (
            "Use a warm, unhurried conversational voice with gentle pitch variation. "
            "Let important words breathe without becoming flat or artificially slow."
        ),
        "text": (
            "Sometimes the best part of a journey is simply slowing down. Take a breath, "
            "look around, and enjoy the small details you might otherwise walk right past."
        ),
    },
    DeliveryStyle.asmr_travel: {
        "direction": (
            "Use a soft, intimate delivery or a comfortable light whisper, with clear words. "
            "Keep the microphone signal usable; avoid clipping, rubbing, or loud breath noise."
        ),
        "text": (
            "Let's stay here for a moment. Listen to the breeze, notice the quiet around us, "
            "and let the rest of the day wait while we take this view in."
        ),
    },
}


def enroll_style(
    workspace: Workspace,
    name: str,
    style: DeliveryStyle,
    source: Path,
    text: str,
    consent: bool,
) -> StyleReference:
    style = DeliveryStyle(style)
    if not consent:
        raise ValueError("Confirm this is your own voice and the transcript is exact.")
    if style == DeliveryStyle.default:
        raise ValueError("Use voice-enroll for the default voice; choose a named delivery style.")
    voice = load_voice(workspace, name)
    if voice.checkpoint:
        raise ValueError("Reference-driven styles require Base voice cloning, not a checkpoint.")
    if style in voice.styles:
        raise ValueError(
            f"Style {style.value!r} already exists; existing recordings are preserved."
        )
    source = source.resolve()
    if not source.is_file() or source.stat().st_size > 250 * 1024 * 1024:
        raise ValueError("Supply a local audio recording no larger than 250 MB.")
    data = decode_audio(source, max_seconds=20)
    quality = audio_quality(data)
    if quality["reasons"]:
        raise ValueError(" ".join(quality["reasons"]))
    directory = workspace.data.audio / "styles" / safe_name(name) / style.value
    if directory.exists():
        raise ValueError("The style audio directory exists; refusing to overwrite recordings.")
    words = len(re.findall(r"\b[\w'-]+\b", text))
    reference = StyleReference(
        audio=str(directory / "reference.wav"),
        text=text,
        words_per_minute=min(210, max(100, words * 60 / quality["seconds"])),
        consent=True,
    )
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="style-", dir=directory.parent) as temporary:
        staged = Path(temporary)
        shutil.copyfile(source, staged / f"raw{source.suffix}")
        write_wav(staged / "reference.wav", data)
        shutil.copytree(staged, directory)
    save_json(
        workspace.data.transcripts / "styles" / name / style.value / "reference.json",
        {
            **reference.model_dump(mode="json"),
            "style": style.value,
            "direction": STYLE_GUIDES[style]["direction"],
            "quality": quality,
            "method": "human-labeled reference audio; no model training or emotion classification",
        },
    )
    voice.styles[style] = reference
    save_json(workspace.voice_path(name), voice)
    return reference


def select_voice(
    workspace: Workspace, name: str, style: DeliveryStyle = DeliveryStyle.default
) -> tuple[Voice, Path]:
    style = DeliveryStyle(style)
    voice = load_voice(workspace, name)
    if style == DeliveryStyle.default:
        return voice, workspace.voice_audio_dir(name)
    if voice.checkpoint:
        raise ValueError("Reference-driven styles require Base voice cloning, not a checkpoint.")
    if style not in voice.styles:
        raise ValueError(
            f"No {style.value!r} reference enrolled for {name!r}. Record your own expressive "
            "3-20 second clip and register it with voice-style-enroll first."
        )
    reference = voice.styles[style]
    audio = workspace.inside(reference.audio)
    if not audio.is_relative_to(workspace.data.audio):
        raise ValueError("Style reference recordings must stay inside the raw audio store.")
    selected = voice.model_copy(
        update={
            "reference_audio": audio.name,
            "reference_text": reference.text,
            "words_per_minute": reference.words_per_minute,
        }
    )
    return selected, audio.parent
