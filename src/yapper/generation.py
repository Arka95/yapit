from __future__ import annotations

import os
import traceback
from pathlib import Path

import numpy as np

from yapit.common_utils.fileutils import save_json
from yapit.common_utils.mediautils import (
    SAMPLE_RATE,
    decode_audio,
    encode_mp3,
    fit_audio,
    write_wav,
)
from yapit.constants import DEFAULT_PROFILE
from yapit.enrollment.styles import select_voice
from yapit.models import Brief, DeliveryStyle, Script, Timeline, Voice
from yapit.storage import OutputArtifacts, Workspace
from yapit.yapper.speech import QwenSpeech, SpeechBackend
from yapit.yapper.timeline import build_timeline


def render_audio(
    script: Script,
    voice: Voice,
    voice_dir: Path,
    job: OutputArtifacts,
    *,
    seconds: float,
    style: DeliveryStyle = DeliveryStyle.default,
    backend: SpeechBackend | None = None,
    cloud_script: bool = False,
) -> tuple[dict, Timeline]:
    script_path = job.path("script.json", job.transcripts)
    save_json(script_path, script)
    save_json(
        job.path("provenance.json"),
        {
            "voice": voice.name,
            "voice_mode": "fine_tuned" if voice.checkpoint else "reference",
            "checkpoint": voice.checkpoint,
            "synthetic_audio": True,
            "cloud_script": cloud_script,
            "delivery_style": style.value,
            "delivery_method": (
                "human-labeled reference audio; listening review required"
                if style != DeliveryStyle.default
                else "default voice reference or checkpoint"
            ),
            "reference_audio": str(voice_dir / voice.reference_audio),
        },
    )
    speech = backend or QwenSpeech(voice, voice_dir)
    pieces = []
    durations = []
    for beat in script.beats:
        wave = np.asarray(speech.synthesize(beat.text), dtype=np.float32).reshape(-1)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError("Speech backend returned empty or invalid audio.")
        pause = np.zeros(round(beat.pause_ms * SAMPLE_RATE / 1000), dtype=np.float32)
        pieces.extend([wave, pause])
        durations.append((len(wave) + len(pause)) / SAMPLE_RATE)
        if sum(durations) > 190:
            raise ValueError("Narration exceeds 190 seconds; shorten the supplied script.")
    raw = job.path("raw.wav", job.audio)
    write_wav(raw, np.concatenate(pieces))
    fitted, factor = fit_audio(raw, job.path("fitted.wav", job.audio), seconds)
    final_wav = job.path("wav", job.audio)
    write_wav(final_wav, fitted)
    actual = len(fitted) / SAMPLE_RATE
    if abs(actual - seconds) > 5:
        raise ValueError("Narration missed the +/-5 second tolerance.")
    final_mp3 = job.path("mp3", job.audio)
    encode_mp3(final_wav, final_mp3)
    encoded = decode_audio(final_mp3)
    actual = len(encoded) / SAMPLE_RATE
    if abs(actual - seconds) > 5:
        raise ValueError("Encoded MP3 missed the +/-5 second tolerance.")
    timeline = build_timeline(script, durations, actual, encoded)
    save_json(job.path("timeline.json"), timeline)
    return {
        "status": "complete",
        "job": str(job.metadata),
        "name": job.name,
        "status_file": str(job.path("status.json")),
        "brief": str(job.path("brief.json")),
        "provenance": str(job.path("provenance.json")),
        "audio": str(final_mp3),
        "wav": str(final_wav),
        "script": str(script_path),
        "video": None,
        "timeline": str(job.path("timeline.json")),
        "captions": None,
        "seconds": actual,
        "tempo_factor": factor,
        "timing_note": timeline.timing_method,
        "delivery_style": style.value,
    }, timeline


def generate_audio(
    script: Script | str,
    *,
    workspace: Workspace | None = None,
    voice_name: str = DEFAULT_PROFILE,
    seconds: float = 30,
    style: DeliveryStyle = DeliveryStyle.default,
    source_name: str | None = None,
    backend: SpeechBackend | None = None,
) -> dict:
    """Speak supplied text unchanged; no script-writing model or video renderer."""
    script = Script.from_text(script) if isinstance(script, str) else script
    if not isinstance(script, Script):
        raise TypeError("Supply plain text or a validated Script.")
    brief = Brief(seconds=seconds)
    style = DeliveryStyle(style)
    root = workspace or Workspace(os.environ.get("YAPIT_WORKSPACE", "workspace"))
    voice, voice_dir = select_voice(root.data, voice_name, style)
    brief = brief.model_copy(update={"words_per_minute": voice.words_per_minute})
    job = root.output(source_name or script.title, voice_name)
    save_json(job.path("brief.json"), brief)
    save_json(job.path("status.json"), {"status": "running", "name": job.name})
    try:
        result, _ = render_audio(
            script, voice, voice_dir, job, seconds=seconds, style=style, backend=backend
        )
        save_json(job.path("status.json"), result)
        return result
    except Exception as exc:
        save_json(
            job.path("status.json"),
            {"status": "failed", "error": str(exc), "error_type": type(exc).__name__},
        )
        job.path("error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
