from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from yapit.common_utils.fileutils import safe_name
from yapit.common_utils.metrics import configure_logging, timed
from yapit.constants import DEFAULT_PROFILE, VIDEO_DURATION_BUFFER
from yapit.editor import AudioMix, generate_video
from yapit.enrollment import enroll
from yapit.models import Brief, DeliveryStyle, Script, Timeline
from yapit.musicgen.planner import plan_music
from yapit.pipeline import Pipeline
from yapit.storage import Workspace
from yapit.yapper import generate_audio


@timed
def create_server(root: Path | None = None) -> FastMCP:
    load_dotenv()
    configure_logging()
    workspace = Workspace(root or os.environ.get("YAPIT_WORKSPACE", "workspace"))
    pipeline = Pipeline(workspace)
    server = FastMCP("Yapit")
    busy = threading.Lock()

    @server.prompt()
    def reel_script(topic: str, seconds: float = 30, kind: str = "travel", facts: str = "") -> str:
        """Reusable non-agentic Instagram reel script prompt."""
        from yapit.yapper.script_planner import build_prompt

        brief = Brief(topic=topic, seconds=seconds, kind=kind, facts=facts)
        return build_prompt(
            brief,
            duration_tolerance=brief.seconds * VIDEO_DURATION_BUFFER,
            end_buffer=AudioMix().end_buffer,
        )

    @server.tool()
    def prepare_script_prompt(
        topic: str = "",
        seconds: float = 30,
        kind: str = "travel",
        facts: str = "",
        media_path: str | None = None,
    ) -> dict:
        """Return a reusable prompt and extracted local frame paths; no cloud requests."""
        media = workspace.inside(media_path) if media_path else None
        return pipeline.prompt(Brief(topic=topic, seconds=seconds, kind=kind, facts=facts), media)

    @server.tool()
    def enroll_voice(name: str, manifest_path: str, consent: bool = True) -> dict:
        """Enroll your own voice; manifest/audio paths must be in workspace or the data root."""
        if not busy.acquire(blocking=False):
            raise ValueError("Yapit is busy; retry after the current job.")
        try:
            voice = enroll(
                workspace.data,
                name,
                workspace.data.inside(manifest_path),
                consent,
                confined=True,
            )
            return {"name": voice.name, "clips": len(voice.samples)}
        finally:
            busy.release()

    def run(
        voice: str,
        seconds: float,
        script: Script | None,
        topic: str,
        kind: str,
        facts: str,
        media_path: str | None,
        auto_script: bool,
        video: bool,
        style: DeliveryStyle,
        music_path: str | None = None,
        mix: AudioMix | None = None,
    ) -> dict:
        if script is not None and auto_script:
            raise ValueError("Choose script or auto_script, not both.")
        if not busy.acquire(blocking=False):
            raise ValueError("Yapit is busy; retry after the current job.")
        try:
            media = workspace.inside(media_path) if media_path else None
            if not video:
                return generate_audio(
                    script,
                    workspace=workspace,
                    voice_name=voice,
                    seconds=seconds,
                    style=style,
                )
            return pipeline.reel(
                Brief(topic=topic, seconds=seconds, kind=kind, facts=facts),
                voice,
                script,
                media,
                auto_script,
                render_video=video,
                style=style,
                music=workspace.inside(music_path) if music_path else None,
                mix=mix,
            )
        finally:
            busy.release()

    @server.tool()
    def generate_reel(
        voice: str = DEFAULT_PROFILE,
        seconds: float = 30,
        script: Script | None = None,
        topic: str = "",
        kind: str = "travel",
        facts: str = "",
        media_path: str | None = None,
        auto_script: bool = False,
        style: DeliveryStyle = DeliveryStyle.default,
        music_path: str | None = None,
        mix: AudioMix | None = None,
    ) -> dict:
        """Generate local MP3, measured timeline and 1080x1920 guide; or return a script prompt."""
        return run(
            voice,
            seconds,
            script,
            topic,
            kind,
            facts,
            media_path,
            auto_script,
            True,
            style,
            music_path,
            mix,
        )

    @server.tool()
    def generate_narration(
        script: Script,
        voice: str = DEFAULT_PROFILE,
        seconds: float = 30,
        style: DeliveryStyle = DeliveryStyle.default,
    ) -> dict:
        """Generate only local cloned-voice MP3 and measured clip timeline."""
        return run(voice, seconds, script, "", "travel", "", None, False, False, style)

    @server.tool()
    def render_timeline(
        timeline_path: str,
        audio_path: str,
        music_path: str | None = None,
        mix: AudioMix | None = None,
    ) -> dict:
        """Render a guide from a timeline/audio inside the workspace or data root."""
        if not busy.acquire(blocking=False):
            raise ValueError("Yapit is busy; retry after the current job.")
        try:
            timeline = Timeline.model_validate_json(
                workspace.inside(timeline_path).read_text(encoding="utf-8"),
            )
            audio = workspace.inside(audio_path)
            if (timeline.width, timeline.height, timeline.fps) != (1080, 1920, 30):
                raise ValueError("Only 1080x1920 at 30 fps is currently supported.")
            music = workspace.inside(music_path) if music_path else None
            settings = mix or AudioMix()
            if music is not None and not settings.beat_music_levels:
                settings = AudioMix.model_validate(
                    {
                        **settings.model_dump(),
                        "beat_music_levels": plan_music(timeline).beat_levels,
                    }
                )
            return generate_video(timeline, audio, workspace=workspace, music=music, mix=settings)
        finally:
            busy.release()

    @server.tool()
    def get_job_status(job_id: str) -> dict:
        """Read status by the returned readable name; legacy job IDs remain readable."""
        if Path(job_id).name != job_id or "/" in job_id or "\\" in job_id:
            raise ValueError("Supply the returned output name, not a path.")
        flat = workspace.root / "outputs" / f"{job_id}.status.json"
        path = workspace.inside(
            flat if flat.is_file() else Path("outputs") / safe_name(job_id) / "status.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    return server


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
