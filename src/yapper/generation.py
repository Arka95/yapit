from __future__ import annotations

import os
import traceback
from contextlib import ExitStack
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
from yapit.common_utils.metrics import timed
from yapit.constants import DEFAULT_PROFILE
from yapit.enrollment.styles import select_voice
from yapit.models import Brief, DeliveryStyle, Script, Timeline, Voice
from yapit.storage import OutputArtifacts, Workspace
from yapit.yapper.backends import SpeechBackend
from yapit.yapper.timeline import build_timeline
from yapit.yapper.worker import backend_for


def _synthesize_chunks(
    script: Script, speech: SpeechBackend, maximum: float
) -> tuple[np.ndarray, list[float]]:
    pieces = []
    durations = []
    total = 0.0
    for beat in script.beats:
        wave = np.asarray(speech.synthesize(beat.text), dtype=np.float32).reshape(-1).copy()
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError("Speech backend returned empty or invalid audio.")
        # Five-millisecond ramps soften joins without trimming speech or changing timing.
        fade_samples = min(SAMPLE_RATE // 200, len(wave) // 2)
        if fade_samples > 1:
            ramp = np.linspace(0, 1, fade_samples, dtype=np.float32)
            wave[:fade_samples] *= ramp
            wave[-fade_samples:] *= ramp[::-1]
        pause = np.zeros(round(beat.pause_ms * SAMPLE_RATE / 1000), dtype=np.float32)
        duration = (len(wave) + len(pause)) / SAMPLE_RATE
        total += duration
        if total > maximum:
            raise ValueError(f"Narration exceeds {maximum:g} seconds; shorten the supplied script.")
        pieces.extend([wave, pause])
        durations.append(duration)
    return np.concatenate(pieces), durations


@timed
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
    natural_duration_tolerance: float | None = None,
    backend_name: str | None = None,
    model: str | None = None,
) -> tuple[dict, Timeline]:
    script_path = job.path("script.json", job.transcripts)
    save_json(script_path, script)
    provenance = {
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
    }
    save_json(job.path("provenance.json"), provenance)
    speech = backend or backend_for(voice, voice_dir, backend_name=backend_name, model=model)
    tolerance = 5 if natural_duration_tolerance is None else natural_duration_tolerance
    maximum = max(190, seconds + tolerance)
    with ExitStack() as resources:
        close = getattr(speech, "close", None)
        if backend is None and callable(close):
            resources.callback(close)
        load = getattr(speech, "load", None)
        if callable(load):
            load()
        wave, durations = _synthesize_chunks(script, speech, maximum)
        describe = getattr(speech, "describe", None)
        provenance.update(describe() if callable(describe) else {"backend": "custom"})
        if provenance["voice_mode"] != "fine_tuned":
            provenance["checkpoint"] = None
    save_json(job.path("provenance.json"), provenance)
    raw = job.path("raw.wav", job.audio)
    write_wav(raw, wave)
    fitted, factor = fit_audio(
        raw,
        job.path("fitted.wav", job.audio),
        seconds,
        natural_duration_tolerance=natural_duration_tolerance,
    )
    final_wav = job.path("wav", job.audio)
    write_wav(final_wav, fitted)
    actual = len(fitted) / SAMPLE_RATE
    if abs(actual - seconds) > tolerance:
        raise ValueError(f"Narration missed the +/-{tolerance:g} second tolerance.")
    final_mp3 = job.path("mp3", job.audio)
    encode_mp3(final_wav, final_mp3)
    encoded = decode_audio(final_mp3, max_seconds=maximum)
    actual = len(encoded) / SAMPLE_RATE
    if abs(actual - seconds) > tolerance:
        raise ValueError(f"Encoded MP3 missed the +/-{tolerance:g} second tolerance.")
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


@timed
def generate_audio(
    script: Script | str,
    *,
    workspace: Workspace | None = None,
    voice_name: str = DEFAULT_PROFILE,
    seconds: float = 30,
    style: DeliveryStyle = DeliveryStyle.default,
    source_name: str | None = None,
    backend: SpeechBackend | None = None,
    backend_name: str | None = None,
    model: str | None = None,
) -> dict:
    """Speak supplied text unchanged; no script-writing model or video renderer."""
    if backend is not None and (backend_name is not None or model is not None):
        raise ValueError("Choose an injected backend or backend/model selection, not both.")
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
            script,
            voice,
            voice_dir,
            job,
            seconds=seconds,
            style=style,
            backend=backend,
            backend_name=backend_name,
            model=model,
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
