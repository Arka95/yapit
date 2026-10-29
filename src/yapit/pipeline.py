from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np

from yapit.media import (
    SAMPLE_RATE,
    decode_audio,
    encode_mp3,
    fit_audio,
    media_frames,
    render_guide,
    write_wav,
)
from yapit.models import Brief, Script
from yapit.scripts import build_prompt, generate_script
from yapit.speech import QwenSpeech, SpeechBackend
from yapit.storage import Workspace, save_json
from yapit.timeline import build_timeline
from yapit.voices import load_voice


class Pipeline:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def prompt(self, brief: Brief, media: Path | None = None) -> dict:
        if not brief.topic and media is None:
            raise ValueError("Provide a topic or a local image/video.")
        job = self.workspace.job()
        frames = media_frames(media, job / "context") if media else []
        text = build_prompt(brief)
        (job / "prompt.txt").write_text(text, encoding="utf-8")
        save_json(job / "brief.json", brief)
        save_json(job / "status.json", {"status": "awaiting_script"})
        return {
            "status": "awaiting_script",
            "job": str(job),
            "prompt": text,
            "frames": [str(frame) for frame in frames],
            "next": "Use this prompt with the frames, then pass the returned Script JSON to reel.",
        }

    def reel(
        self,
        brief: Brief,
        voice_name: str,
        script: Script | None = None,
        media: Path | None = None,
        auto_script: bool = False,
        backend: SpeechBackend | None = None,
        render_video: bool = True,
    ) -> dict:
        if script is None and not auto_script:
            return self.prompt(brief, media)
        voice = load_voice(self.workspace, voice_name)
        brief = brief.model_copy(update={"words_per_minute": voice.words_per_minute})
        job = self.workspace.job()
        save_json(job / "brief.json", brief)
        save_json(job / "status.json", {"status": "running"})
        try:
            cloud_script = script is None and auto_script
            frames = media_frames(media, job / "context") if media else []
            if script is None:
                script = generate_script(brief, frames)
            save_json(job / "script.json", script)
            save_json(
                job / "provenance.json",
                {
                    "voice": voice_name,
                    "voice_mode": "fine_tuned" if voice.checkpoint else "reference",
                    "checkpoint": voice.checkpoint,
                    "synthetic_audio": True,
                    "cloud_script": cloud_script,
                },
            )
            speech = backend or QwenSpeech(voice, self.workspace.voice_path(voice_name).parent)
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
            raw = job / "raw.wav"
            write_wav(raw, np.concatenate(pieces))
            fitted, factor = fit_audio(raw, job / "fitted.wav", brief.seconds)
            final_wav = job / "narration.wav"
            write_wav(final_wav, fitted)
            seconds = len(fitted) / SAMPLE_RATE
            if abs(seconds - brief.seconds) > 5:
                raise ValueError("Narration missed the +/-5 second tolerance.")
            encode_mp3(final_wav, job / "narration.mp3")
            encoded = decode_audio(job / "narration.mp3")
            seconds = len(encoded) / SAMPLE_RATE
            if abs(seconds - brief.seconds) > 5:
                raise ValueError("Encoded MP3 missed the +/-5 second tolerance.")
            timeline = build_timeline(script, durations, seconds, encoded)
            save_json(job / "timeline.json", timeline)
            if render_video:
                render_guide(timeline, job / "narration.mp3", job / "video")
            result = {
                "status": "complete",
                "job": str(job),
                "audio": str(job / "narration.mp3"),
                "video": str(job / "video" / "guide.mp4") if render_video else None,
                "timeline": str(job / "timeline.json"),
                "captions": str(job / "video" / "captions.srt") if render_video else None,
                "seconds": seconds,
                "tempo_factor": factor,
                "timing_note": timeline.timing_method,
            }
            save_json(job / "status.json", result)
            return result
        except Exception as exc:
            save_json(
                job / "status.json",
                {
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            (job / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            raise
