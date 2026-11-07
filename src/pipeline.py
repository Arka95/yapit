from __future__ import annotations

import traceback
from pathlib import Path

from yapit.common_utils.fileutils import save_json
from yapit.common_utils.mediautils import media_frames
from yapit.constants import DEFAULT_PROFILE
from yapit.editor import render_guide
from yapit.enrollment.styles import select_voice
from yapit.models import Brief, DeliveryStyle, Script
from yapit.storage import Workspace
from yapit.yapper import SpeechBackend
from yapit.yapper.generation import render_audio
from yapit.yapper.script_planner import build_prompt, generate_script


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
        voice, voice_dir = select_voice(self.workspace.data, voice_name, style)
        brief = brief.model_copy(update={"words_per_minute": voice.words_per_minute})
        source = source_name or (media.stem if media else script.title if script else brief.topic)
        job = self.workspace.output(source, voice_name)
        save_json(job.path("brief.json"), brief)
        save_json(job.path("status.json"), {"status": "running", "name": job.name})
        try:
            transcript_dir = job.transcripts
            cloud_script = script is None and auto_script
            frames = media_frames(media, job.metadata, prefix=job.name) if media else []
            if script is None:
                script = generate_script(brief, frames)
            result, timeline = render_audio(
                script,
                voice,
                voice_dir,
                job,
                seconds=brief.seconds,
                style=style,
                backend=backend,
                cloud_script=cloud_script,
            )
            if render_video:
                render_guide(
                    timeline,
                    Path(result["audio"]),
                    job.metadata,
                    video=job.path("guide.mp4", job.video),
                    captions=job.path("captions.srt", transcript_dir),
                    prefix=job.name,
                )
            result.update(
                {
                    "video": str(job.path("guide.mp4", job.video)) if render_video else None,
                    "captions": str(job.path("captions.srt", transcript_dir))
                    if render_video
                    else None,
                }
            )
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
