from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np

from yapit.common_utils.fileutils import save_json
from yapit.common_utils.mediautils import (
    SAMPLE_RATE,
    decode_audio,
    encode_mp3,
    fit_audio,
    media_frames,
    write_wav,
)
from yapit.constants import DEFAULT_PROFILE
from yapit.media import render_guide
from yapit.models import Brief, DeliveryStyle, Script
from yapit.scripts import build_prompt, generate_script
from yapit.speech import QwenSpeech, SpeechBackend
from yapit.storage import Workspace
from yapit.timeline import build_timeline
from yapit.voices import select_voice


class Pipeline:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def prompt(
        self, brief: Brief, media: Path | None = None, profile: str = DEFAULT_PROFILE
    ) -> dict:
        if not brief.topic and media is None:
            raise ValueError("Provide a topic or a local image/video.")
        job = self.workspace.output(media.stem if media else brief.topic, profile)
        frames = media_frames(media, job.metadata, prefix=job.name) if media else []
        text = build_prompt(brief)
        job.path("prompt.txt").write_text(text, encoding="utf-8")
        save_json(job.path("brief.json"), brief)
        save_json(job.path("status.json"), {"status": "awaiting_script", "name": job.name})
        return {
            "status": "awaiting_script",
            "job": str(job.metadata),
            "name": job.name,
            "status_file": str(job.path("status.json")),
            "brief": str(job.path("brief.json")),
            "prompt_file": str(job.path("prompt.txt")),
            "prompt": text,
            "frames": [str(frame) for frame in frames],
            "next": "Use this prompt with the frames, then pass the returned Script JSON to reel.",
        }

    def reel(
        self,
        brief: Brief,
        voice_name: str = DEFAULT_PROFILE,
        script: Script | None = None,
        media: Path | None = None,
        auto_script: bool = False,
        backend: SpeechBackend | None = None,
        render_video: bool = True,
        style: DeliveryStyle = DeliveryStyle.default,
        source_name: str | None = None,
    ) -> dict:
        style = DeliveryStyle(style)
        if script is None and not auto_script:
            return self.prompt(brief, media, voice_name)
        voice, voice_dir = select_voice(self.workspace, voice_name, style)
        brief = brief.model_copy(update={"words_per_minute": voice.words_per_minute})
        source = source_name or (media.stem if media else script.title if script else brief.topic)
        job = self.workspace.output(source, voice_name)
        save_json(job.path("brief.json"), brief)
        save_json(job.path("status.json"), {"status": "running", "name": job.name})
        try:
            transcript_dir = job.transcripts
            audio_dir = job.audio
            cloud_script = script is None and auto_script
            frames = media_frames(media, job.metadata, prefix=job.name) if media else []
            if script is None:
                script = generate_script(brief, frames)
            script_path = job.path("script.json", transcript_dir)
            save_json(script_path, script)
            save_json(
                job.path("provenance.json"),
                {
                    "voice": voice_name,
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
                wave = speech.synthesize(beat.text)
                wave = np.asarray(wave, dtype=np.float32).reshape(-1)
                if not len(wave) or not np.isfinite(wave).all():
                    raise ValueError("Speech backend returned empty or invalid audio.")
                pause = np.zeros(round(beat.pause_ms * SAMPLE_RATE / 1000), dtype=np.float32)
                pieces.extend([wave, pause])
                durations.append((len(wave) + len(pause)) / SAMPLE_RATE)
                if sum(durations) > 190:
                    raise ValueError("Narration exceeds 190 seconds; shorten the supplied script.")
            raw = job.path("raw.wav", audio_dir)
            write_wav(raw, np.concatenate(pieces))
            fitted, factor = fit_audio(raw, job.path("fitted.wav", audio_dir), brief.seconds)
            final_wav = job.path("wav", audio_dir)
            write_wav(final_wav, fitted)
            seconds = len(fitted) / SAMPLE_RATE
            if abs(seconds - brief.seconds) > 5:
                raise ValueError("Narration missed the +/-5 second tolerance.")
            final_mp3 = job.path("mp3", audio_dir)
            encode_mp3(final_wav, final_mp3)
            encoded = decode_audio(final_mp3)
            seconds = len(encoded) / SAMPLE_RATE
            if abs(seconds - brief.seconds) > 5:
                raise ValueError("Encoded MP3 missed the +/-5 second tolerance.")
            timeline = build_timeline(script, durations, seconds, encoded)
            save_json(job.path("timeline.json"), timeline)
            if render_video:
                render_guide(
                    timeline,
                    final_mp3,
                    job.metadata,
                    video=job.path("guide.mp4", job.video),
                    captions=job.path("captions.srt", transcript_dir),
                    prefix=job.name,
                )
            result = {
                "status": "complete",
                "job": str(job.metadata),
                "name": job.name,
                "status_file": str(job.path("status.json")),
                "brief": str(job.path("brief.json")),
                "provenance": str(job.path("provenance.json")),
                "audio": str(final_mp3),
                "wav": str(final_wav),
                "script": str(script_path),
                "video": str(job.path("guide.mp4", job.video)) if render_video else None,
                "timeline": str(job.path("timeline.json")),
                "captions": str(job.path("captions.srt", transcript_dir)) if render_video else None,
                "seconds": seconds,
                "tempo_factor": factor,
                "timing_note": timeline.timing_method,
                "delivery_style": style.value,
            }
            save_json(job.path("status.json"), result)
            return result
        except Exception as exc:
            save_json(
                job.path("status.json"),
                {
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            job.path("error.log").write_text(traceback.format_exc(), encoding="utf-8")
            raise
