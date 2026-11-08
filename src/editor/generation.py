from __future__ import annotations

import os
from pathlib import Path

from yapit.common_utils.fileutils import save_json
from yapit.common_utils.metrics import timed
from yapit.editor.mixing import AudioMix
from yapit.editor.render import RenderSettings, render_guide
from yapit.editor.templates import DEFAULT_STYLE, PlaceholderStyle
from yapit.models import Timeline
from yapit.storage import Workspace


@timed
def generate_video(
    timeline: Timeline,
    audio: Path,
    *,
    workspace: Workspace | None = None,
    music: Path | None = None,
    mix: AudioMix | None = None,
    render_settings: RenderSettings | None = None,
    style: PlaceholderStyle = DEFAULT_STYLE,
    actual_seconds: float | None = None,
    source_name: str | None = None,
) -> dict:
    """Render existing narration without TTS, music search or any hosted provider."""
    root = workspace or Workspace(os.environ.get("YAPIT_WORKSPACE", "workspace"))
    settings = mix or AudioMix()
    encoding = render_settings or RenderSettings()
    job = root.output(source_name or audio.stem)
    timeline_path = job.path("timeline.json")
    mix_path = job.path("mix.json")
    status_path = job.path("status.json")
    video_path = job.path("guide.mp4", job.video)
    captions_path = job.path("captions.srt", job.transcripts)
    save_json(timeline_path, timeline)
    save_json(
        mix_path,
        {
            "audio": str(audio.resolve()),
            "music": str(music.resolve()) if music else None,
            "settings": settings.model_dump(mode="json"),
            "encoding": encoding.model_dump(mode="json"),
            "fades_apply_to": "music_only",
        },
    )
    try:
        seconds = render_guide(
            timeline,
            audio,
            job.metadata,
            video=video_path,
            captions=captions_path,
            prefix=job.name,
            music=music,
            mix=settings,
            render_settings=encoding,
            style=style,
            actual_seconds=actual_seconds,
        )
        result = {
            "status": "complete",
            "name": job.name,
            "status_file": str(status_path),
            "video": str(video_path),
            "captions": str(captions_path),
            "audio": str(audio.resolve()),
            "music": str(music.resolve()) if music else None,
            "mix": str(mix_path),
            "timeline": str(timeline_path),
            "seconds": timeline.seconds,
            "video_seconds": seconds,
        }
        save_json(status_path, result)
        return result
    except Exception as exc:
        save_json(status_path, {"status": "failed", "error": str(exc)})
        raise
