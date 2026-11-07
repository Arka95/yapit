from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from yapit.common_utils.fileutils import safe_name, save_json
from yapit.common_utils.mediautils import SAMPLE_RATE, decode_audio
from yapit.constants import DEFAULT_PROFILE
from yapit.enrollment import enroll
from yapit.media import render_guide
from yapit.models import Brief, DeliveryStyle, Script, Timeline
from yapit.pipeline import Pipeline
from yapit.storage import Workspace


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
                style=style,
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
    ) -> dict:
        """Generate local MP3, measured timeline and 1080x1920 guide; or return a script prompt."""
        return run(voice, seconds, script, topic, kind, facts, media_path, auto_script, True, style)

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
    def render_timeline(timeline_path: str, audio_path: str) -> dict:
        """Render a guide from a timeline/audio inside the workspace or data root."""
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
            job = workspace.output(audio.stem)
            captions = job.path("captions.srt", job.transcripts)
            try:
                render_guide(timeline, audio, job.metadata, captions=captions, prefix=job.name)
                result = {
                    "status": "complete",
                    "name": job.name,
                    "status_file": str(job.path("status.json")),
                    "video": str(job.path("guide.mp4")),
                    "captions": str(captions),
                }
                save_json(job.path("status.json"), result)
                return result
            except Exception as exc:
                save_json(job.path("status.json"), {"status": "failed", "error": str(exc)})
                raise
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
