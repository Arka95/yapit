from __future__ import annotations

import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from yapit import enrollment, pipeline, rescript, yapper
from yapit.common_utils.metrics import configure_logging, timed
from yapit.constants import DEFAULT_ASR_MODEL, DEFAULT_PROFILE
from yapit.editor import AudioMix
from yapit.models import DeliveryStyle, Script
from yapit.storage import Workspace


@timed
def create_server(root: Path | None = None) -> FastMCP:
    load_dotenv()
    configure_logging()
    workspace = Workspace(root or os.environ.get("YAPIT_WORKSPACE", "workspace"))
    server = FastMCP("Yapit")
    busy = threading.Lock()

    @server.tool()
    def generate_script(prompt: str, seconds: float = 30) -> dict:
        """Write a new narration script from a text request.

        prompt describes the subject, instructions, and facts to use. seconds is
        the target duration, default 30 (range 5-180), not a guaranteed speaking time.
        Returns Script JSON with title and beats containing text, visual, and
        pause_ms. No images, frames, media paths, audio, or video are accepted.
        Requires YAPIT_ALLOW_CLOUD=true, OPENAI_API_KEY, and the .[cloud] dependency.
        To speak the result, pass it as script to generate_audio or generate_reel.
        """
        if not busy.acquire(blocking=False):
            raise ValueError("Yapit is busy; retry after the current job.")
        try:
            return yapper.generate_script(prompt, seconds=seconds).model_dump(mode="json")
        finally:
            busy.release()

    @server.tool()
    def enroll_voice() -> dict:
        """Open the local voice-enrollment app; no inputs or files are required.

        In the app, choose or create a profile, select a microphone, record the
        displayed passages, listen back and finish enrollment. The app manages
        recordings and transcripts; you do not supply an MP3 or manifest.
        Returns a launch receipt and process ID immediately, not a finished
        profile. Complete enrollment in the window before generating audio.
        Requires an interactive Windows desktop on the server machine.
        No speech recognition, cloud calls or model training runs.
        """
        if not busy.acquire(blocking=False):
            raise ValueError("Yapit is busy; retry after the current job.")
        try:
            return enrollment.enroll_voice(workspace=workspace)
        finally:
            busy.release()

    @server.tool()
    def extract_script(
        media_path: str,
        model: str = DEFAULT_ASR_MODEL,
        language: str = "en",
        offline: bool = True,
    ) -> dict:
        """Extract reviewable YapIt Script JSON from a local audio or video file.

        media_path must be inside the configured workspace/data roots. Uses local
        speech recognition, with cached small.en and English by default; offline
        defaults to true. Set offline=false only to permit model downloads.
        Returns transcript/script paths, source timings, and actual Script JSON in
        scripts[*].content. Longer recordings can produce multiple scripts.
        Recognized words require review, especially names; no cloud rewriting, voice
        enrollment or speech generation is performed.
        """
        if not busy.acquire(blocking=False):
            raise ValueError("Yapit is busy; retry after the current job.")
        try:
            return rescript.extract_script(
                workspace.inside(media_path),
                model=model,
                language=language,
                offline=offline,
                data=workspace.data,
            )
        finally:
            busy.release()

    @server.tool()
    def generate_reel(
        script: Script | None = None,
        audio_path: str | None = None,
        prompt: str | None = None,
        voice: str = DEFAULT_PROFILE,
        seconds: float = 30,
        style: DeliveryStyle = DeliveryStyle.default,
        music_path: str | None = None,
        mix: AudioMix | None = None,
    ) -> dict:
        """Create a portrait edit guide from exactly one script, audio_path or prompt.

        script is approved YapIt Script JSON; its words are spoken unchanged.
        prompt is text for the full script -> narration -> video pipeline and
        requires YAPIT_ALLOW_CLOUD=true, OPENAI_API_KEY and the cloud dependency.
        audio_path preserves existing audio: offline local speech recognition builds
        timing and captions, without a supplied timeline, enrolled voice or new speech.
        Audio input must be at most 180 seconds; its duration is preserved, so
        seconds and voice apply only to script/prompt input (defaults 30/testuser).
        File inputs, including optional music_path, must be inside workspace/data.
        Returns audio, timeline, captions, status and a 1080x1920 placeholder video,
        not finished footage. Editor owns visual templates; no images are written
        by the script generator. Calls finish synchronously, not in a queue.
        """
        if not busy.acquire(blocking=False):
            raise ValueError("Yapit is busy; retry after the current job.")
        try:
            return pipeline.generate_reel(
                script,
                audio_path=workspace.inside(audio_path) if audio_path is not None else None,
                prompt=prompt,
                voice=voice,
                seconds=seconds,
                workspace=workspace,
                style=style,
                music=workspace.inside(music_path) if music_path else None,
                mix=mix,
            )
        finally:
            busy.release()

    @server.tool()
    def generate_audio(
        script: Script,
        voice: str = DEFAULT_PROFILE,
        seconds: float = 30,
        style: DeliveryStyle = DeliveryStyle.default,
    ) -> dict:
        """Speak an approved script using a local voice profile, without creating video.

        script is YapIt Script JSON with title and beats, not a file path or a
        writing prompt. voice defaults to the enrolled testuser profile; seconds
        defaults to 30. The target permits +/-5 seconds and bounded tempo fitting,
        so substantially mismatched scripts fail rather than losing words.
        Named styles require a recorded reference on that profile.
        Returns MP3/WAV, measured timeline, saved script, and provenance/status paths.
        Supplied words are not rewritten; no script-writing cloud request is made.
        """
        if not busy.acquire(blocking=False):
            raise ValueError("Yapit is busy; retry after the current job.")
        try:
            return yapper.generate_audio(
                script, voice_name=voice, seconds=seconds, workspace=workspace, style=style
            )
        finally:
            busy.release()

    return server


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
