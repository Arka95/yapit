from __future__ import annotations

import os
from pathlib import Path

from yapit.common_utils.fileutils import save_json
from yapit.editor.mixing import AudioMix
from yapit.editor.render import render_guide
from yapit.models import Timeline
from yapit.storage import Workspace


def generate_video(
    timeline: Timeline,
    audio: Path,
    *,
    workspace: Workspace | None = None,
    music: Path | None = None,
    mix: AudioMix | None = None,
    source_name: str | None = None,
) -> dict:
    """Render existing narration without TTS, music search or any hosted provider."""
    root = workspace or Workspace(os.environ.get("YAPIT_WORKSPACE", "workspace"))
    settings = mix or AudioMix()
    job = root.output(source_name or audio.stem)
    save_json(job.path("timeline.json"), timeline)
    save_json(
        job.path("mix.json"),
        {
            "audio": str(audio.resolve()),
            "music": str(music.resolve()) if music else None,
            "settings": settings.model_dump(mode="json"),
            "fades_apply_to": "music_only",
        },
    )
    try:
        seconds = render_guide(
            timeline,
            audio,
            job.metadata,
            video=job.path("guide.mp4", job.video),
            captions=job.path("captions.srt", job.transcripts),
            prefix=job.name,
            music=music,
            mix=settings,
        )
        result = {
            "status": "complete",
            "name": job.name,
            "status_file": str(job.path("status.json")),
            "video": str(job.path("guide.mp4", job.video)),
            "captions": str(job.path("captions.srt", job.transcripts)),
            "audio": str(audio.resolve()),
            "music": str(music.resolve()) if music else None,
            "mix": str(job.path("mix.json")),
            "timeline": str(job.path("timeline.json")),
            "seconds": timeline.seconds,
            "video_seconds": seconds,
        }
        save_json(job.path("status.json"), result)
        return result
    except Exception as exc:
        save_json(job.path("status.json"), {"status": "failed", "error": str(exc)})
        raise
