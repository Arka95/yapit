from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from yapit import pipeline, rescript
from yapit.common_utils.mediautils import ffmpeg
from yapit.common_utils.metrics import configure_logging
from yapit.common_utils.runtime_config import set_runtime_mode
from yapit.constants import DEFAULT_ASR_MODEL, DEFAULT_PROFILE, DEFAULT_SPEECH_MODEL
from yapit.editor import AudioMix
from yapit.enrollment.cli import data_store
from yapit.enrollment.cli import enroll_voice as enroll_voice_command
from yapit.enrollment.cli import profiles as profiles_command
from yapit.enrollment.styles import STYLE_GUIDES, enroll_style
from yapit.models import DeliveryStyle
from yapit.musicgen.cli import music_download, music_search
from yapit.musicgen.cli import music_plan as music_plan_command
from yapit.rescript.cli import extract_audio
from yapit.storage import Workspace
from yapit.training import resume_checkpoint, select_checkpoint, train
from yapit.yapper import generate_script as write_script
from yapit.yapper.cli import generate as generate_audio_command
from yapit.yapper.script_planner import load_script

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
PathOption = Annotated[Path | None, typer.Option()]

app.command("generate-audio")(generate_audio_command)
app.command("extract_audio")(extract_audio)
app.command("profiles")(profiles_command)
app.command("enroll-voice")(enroll_voice_command)
app.command("music-plan")(music_plan_command)
app.command("music-search")(music_search)
app.command("music-download")(music_download)


def workspace() -> Workspace:
    load_dotenv()
    configure_logging()
    return Workspace(os.environ.get("YAPIT_WORKSPACE", "workspace"))


def emit(value: dict) -> None:
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False))


@app.command("set-mode")
def set_mode(
    mode: Annotated[str, typer.Argument(help="cpu or gpu; saved in the local .env.")],
) -> None:
    """Choose the runtime for future yapit.cmd calls without installing or loading models."""
    try:
        emit(set_runtime_mode(Path.cwd(), mode))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def doctor() -> None:
    """Report dependencies; never download speech models."""
    root = workspace()
    report = {
        "workspace": str(root.root),
        "data": str(root.data.root),
        "ffmpeg": ffmpeg(),
        "dependencies": {
            module: importlib.util.find_spec(module) is not None
            for module in ["qwen_tts", "torch", "mcp", "openai"]
        },
        "cloud_allowed": os.environ.get("YAPIT_ALLOW_CLOUD", "").lower() == "true",
        "openai_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
    }
    if report["dependencies"]["torch"]:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
    emit(report)


@app.command("voice-worker")
def voice_worker(
    voice: Annotated[str, typer.Argument()] = DEFAULT_PROFILE,
    backend: str | None = None,
    model: str | None = None,
) -> None:
    """Keep one local speech model loaded; cache unchanged speech until explicitly stopped."""
    from yapit.yapper.worker import run_worker

    run_worker(data_store(), voice, backend=backend, model=model)


@app.command("voice-worker-status")
def voice_worker_status() -> None:
    """Show worker readiness, actual device and load/generation timings."""
    from yapit.yapper.worker import worker_status

    emit(worker_status(data_store()))


@app.command("voice-worker-stop")
def voice_worker_stop() -> None:
    """Ask the local worker to stop and release its model/GPU memory."""
    from yapit.yapper.worker import stop_worker

    emit(stop_worker(data_store()))


@app.command("voice-style-prompt")
def voice_style_prompt(style: DeliveryStyle = DeliveryStyle.lively_vlog) -> None:
    """Get a short passage and direction for recording your own expressive reference."""
    if style == DeliveryStyle.default:
        raise typer.BadParameter("Choose a named style, not default.")
    emit(
        {
            "style": style.value,
            **STYLE_GUIDES[style],
            "recording_seconds": "3-20; aim for 8-15",
            "next": "Record your own voice, verify the exact text, then use enroll-style.",
        }
    )


@app.command("enroll-style")
def enroll_style_command(
    name: str,
    audio: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    text: Annotated[str, typer.Option(help="Exact manually verified words in the recording.")],
    style: DeliveryStyle = DeliveryStyle.lively_vlog,
    consent: Annotated[bool, typer.Option(hidden=True)] = True,
) -> None:
    """Register an expressive recording for an existing voice, without training."""
    try:
        reference = enroll_style(workspace().data, name, style, audio, text, consent)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    emit({"voice": name, "style": style.value, **reference.model_dump(mode="json")})


@app.command()
def generate_script(
    prompt: Annotated[str, typer.Argument(help="Text prompt for the narration script.")],
    seconds: float = 30,
) -> None:
    """Generate YapIt Script JSON from a text prompt; requires explicit cloud opt-in."""
    load_dotenv()
    configure_logging()
    try:
        result = write_script(prompt, seconds=seconds)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    emit(result.model_dump(mode="json"))


@app.command("extract-script")
def extract_script(
    media_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    model: str = DEFAULT_ASR_MODEL,
    language: str = "en",
    offline: bool = True,
) -> None:
    """Extract reviewable Script JSON from local audio/video; offline by default."""
    root = workspace()
    try:
        emit(
            rescript.extract_script(
                media_path, model=model, language=language, offline=offline, data=root.data
            )
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("generate-reel")
def generate_reel(
    script: PathOption = None,
    audio_path: PathOption = None,
    prompt: str | None = None,
    voice: str = DEFAULT_PROFILE,
    seconds: float = 30,
    style: DeliveryStyle = DeliveryStyle.default,
    music: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    audio_volume: float = 100,
    music_volume: float = 20,
    music_fade_in: float = 0.5,
    music_fade_out: float = 1,
    duck_music: bool = True,
    loop_music: bool = True,
    end_buffer: float = 0.75,
) -> None:
    """Create an edit guide from exactly one --script, --audio-path or --prompt."""
    root = workspace()
    try:
        if sum(value is not None for value in (script, audio_path, prompt)) != 1:
            raise ValueError("Choose exactly one of --script, --audio-path or --prompt.")
        emit(
            pipeline.generate_reel(
                load_script(script) if script is not None else None,
                audio_path=audio_path,
                prompt=prompt,
                voice=voice,
                seconds=seconds,
                workspace=root,
                style=style,
                music=music,
                mix=AudioMix(
                    audio_volume=audio_volume,
                    music_volume=music_volume,
                    music_fade_in=music_fade_in,
                    music_fade_out=music_fade_out,
                    duck_music=duck_music,
                    loop_music=loop_music,
                    end_buffer=end_buffer,
                ),
            )
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("voice-train")
def voice_train(
    name: str,
    upstream: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    epochs: int = 3,
    batch_size: int = 1,
    model: Annotated[str, typer.Option(help="Official 1.7B or 0.6B Base Hub model ID.")] = (
        DEFAULT_SPEECH_MODEL
    ),
    resume: Annotated[
        Path | None,
        typer.Option(
            exists=True, file_okay=False, help="Continue checkpoint weights; new optimizer."
        ),
    ] = None,
) -> None:
    """Fine-tune locally using a reviewed official Qwen3-TTS checkout (Linux/CUDA)."""
    root = workspace()
    emit(
        resume_checkpoint(
            root, name, resume, upstream, epochs=epochs, batch_size=batch_size, model=model
        )
        if resume is not None
        else train(root, name, upstream, epochs=epochs, batch_size=batch_size, model=model)
    )


@app.command("voice-select-checkpoint")
def voice_select_checkpoint(
    name: str,
    checkpoint: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    """Select a fine-tuned checkpoint after listening review."""
    select_checkpoint(workspace(), name, checkpoint)
    emit({"voice": name, "checkpoint": str(checkpoint.resolve())})


if __name__ == "__main__":
    app()
