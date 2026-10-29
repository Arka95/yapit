from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from yapit.media import SAMPLE_RATE, decode_audio, render_guide
from yapit.models import Brief, Script, Timeline
from yapit.pipeline import Pipeline
from yapit.storage import Workspace, safe_name
from yapit.voices import enroll


def create_server(root: Path | None = None) -> FastMCP:
    load_dotenv()
    workspace = Workspace(root or os.environ.get("YAPIT_WORKSPACE", "workspace"))
    pipeline = Pipeline(workspace)
    server = FastMCP("Yapit")
    busy = threading.Lock()

    @server.prompt()
    def reel_script(topic: str, seconds: float = 30, kind: str = "travel", facts: str = "") -> str:
        """Reusable non-agentic Instagram reel script prompt."""
        from yapit.scripts import build_prompt

        return build_prompt(Brief(topic=topic, seconds=seconds, kind=kind, facts=facts))

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
    def enroll_voice(name: str, manifest_path: str, consent: bool = False) -> dict:
        """Enroll your own consenting voice. Manifest and every audio path must be in workspace."""
        if not busy.acquire(blocking=False):
            raise ValueError("Yapit is busy; retry after the current job.")
        try:
            voice = enroll(
                workspace,
                name,
                workspace.inside(manifest_path),
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
    ) -> dict:
        if script is not None and auto_script:
            raise ValueError("Choose script or auto_script, not both.")
        if not busy.acquire(blocking=False):
            raise ValueError("Yapit is busy; retry after the current job.")
        try:
            media = workspace.inside(media_path) if media_path else None
            return pipeline.reel(
                Brief(topic=topic, seconds=seconds, kind=kind, facts=facts),
                voice,
                script,
                media,
                auto_script,
                render_video=video,
            )
        finally:
            busy.release()

    @server.tool()
    def generate_reel(
        voice: str,
        seconds: float = 30,
        script: Script | None = None,
        topic: str = "",
        kind: str = "travel",
        facts: str = "",
        media_path: str | None = None,
        auto_script: bool = False,
    ) -> dict:
        """Generate local MP3, measured timeline and 1080x1920 guide; or return a script prompt."""
        return run(voice, seconds, script, topic, kind, facts, media_path, auto_script, True)

    @server.tool()
    def generate_narration(voice: str, script: Script, seconds: float = 30) -> dict:
        """Generate only local cloned-voice MP3 and measured clip timeline."""
        return run(voice, seconds, script, "", "travel", "", None, False, False)

    @server.tool()
    def render_timeline(timeline_path: str, audio_path: str) -> dict:
        """Render a guide from an existing timeline and matching narration inside workspace."""
        if not busy.acquire(blocking=False):
            raise ValueError("Yapit is busy; retry after the current job.")
        try:
            timeline = Timeline.model_validate_json(
                workspace.inside(timeline_path).read_text(encoding="utf-8"),
            )
            audio = workspace.inside(audio_path)
            actual = len(decode_audio(audio)) / SAMPLE_RATE
            if abs(actual - timeline.seconds) > 0.1:
                raise ValueError("Timeline and audio duration differ by more than 100 ms.")
            if (timeline.width, timeline.height, timeline.fps) != (1080, 1920, 30):
                raise ValueError("Only 1080x1920 at 30 fps is currently supported.")
            job = workspace.job()
            render_guide(timeline, audio, job / "video")
            return {"video": str(job / "video" / "guide.mp4")}
        finally:
            busy.release()

    @server.tool()
    def get_job_status(job_id: str) -> dict:
        """Read a persisted job status; job_id is the output directory's basename."""
        path = workspace.inside(Path("outputs") / safe_name(job_id) / "status.json")
        return json.loads(path.read_text(encoding="utf-8"))

    return server


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
