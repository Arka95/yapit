from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from yapit.common_utils.mediautils import ffmpeg
from yapit.constants import DEFAULT_PROFILE
from yapit.enrollment import enroll
from yapit.enrollment.cli import data_store
from yapit.models import Brief, DeliveryStyle
from yapit.pipeline import Pipeline
from yapit.rescript.cli import app as rescript_app
from yapit.scripts import load_script
from yapit.storage import Workspace
from yapit.training import train, use_checkpoint
from yapit.voices import STYLE_GUIDES, enroll_style, select_voice

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
PathOption = Annotated[Path | None, typer.Option()]

app.add_typer(rescript_app, name="rescript")


def workspace() -> Workspace:
    load_dotenv()
    return Workspace(os.environ.get("YAPIT_WORKSPACE", "workspace"))


def emit(value: dict) -> None:
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False))


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


@app.command("voice-enroll")
def voice_enroll(
    name: str,
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    consent: Annotated[bool, typer.Option(hidden=True)] = True,
) -> None:
    """Curate local recordings with manually verified transcripts."""
    try:
        voice = enroll(data_store(), name, manifest, consent)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    emit(
        {
            "name": voice.name,
            "clips": len(voice.samples),
            "words_per_minute": voice.words_per_minute,
        }
    )


@app.command("enroll-app")
def enroll_app() -> None:
    """Open the native Windows microphone enrollment wizard."""
    from yapit.enrollment.desktop import main

    main()


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
            "next": "Record your own voice, verify the exact text, then use voice-style-enroll.",
        }
    )


@app.command("voice-style-enroll")
def voice_style_enroll(
    name: str,
    audio: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    text: Annotated[str, typer.Option(help="Exact manually verified words in the recording.")],
    style: DeliveryStyle = DeliveryStyle.lively_vlog,
    consent: Annotated[
        bool, typer.Option(help="This is my own voice, and I verified the exact transcript.")
    ] = False,
) -> None:
    """Register an expressive recording for an existing voice, without training."""
    try:
        reference = enroll_style(workspace(), name, style, audio, text, consent)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    emit({"voice": name, "style": style.value, **reference.model_dump(mode="json")})


@app.command()
def prompt(
    topic: str = "",
    seconds: float = 30,
    kind: str = "travel",
    facts: str = "",
    media: PathOption = None,
) -> None:
    """Create a reusable script prompt plus local visual context."""
    brief = Brief(topic=topic, seconds=seconds, kind=kind, facts=facts)
    emit(Pipeline(workspace()).prompt(brief, media))


@app.command()
def reel(
    voice: Annotated[str, typer.Argument()] = DEFAULT_PROFILE,
    seconds: float = 30,
    script: PathOption = None,
    topic: str = "",
    kind: str = "travel",
    facts: str = "",
    media: PathOption = None,
    auto_script: bool = False,
    style: DeliveryStyle = DeliveryStyle.default,
) -> None:
    """Orchestrate script -> local narration -> 1080x1920 edit guide."""
    root = workspace()
    if script is not None and auto_script:
        raise typer.BadParameter("Choose --script or --auto-script, not both.")
    rate = select_voice(root, voice, style)[0].words_per_minute if script or auto_script else 150
    brief = Brief(topic=topic, seconds=seconds, kind=kind, facts=facts, words_per_minute=rate)
    result = Pipeline(root).reel(
        brief,
        voice,
        load_script(script) if script else None,
        media,
        auto_script,
        style=style,
        source_name=media.stem if media else script.stem if script else None,
    )
    emit(result)


@app.command()
def narrate(
    script: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    voice: Annotated[str, typer.Argument()] = DEFAULT_PROFILE,
    seconds: float = 30,
    style: DeliveryStyle = DeliveryStyle.default,
) -> None:
    """Render only narration MP3 and the measured clip timeline."""
    root = workspace()
    emit(
        Pipeline(root).reel(
            Brief(seconds=seconds),
            voice,
            load_script(script),
            render_video=False,
            style=style,
            source_name=script.stem,
        )
    )


@app.command("voice-train")
def voice_train(
    name: str,
    upstream: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    epochs: int = 3,
    batch_size: int = 1,
) -> None:
    """Fine-tune locally using a reviewed official Qwen3-TTS checkout (Linux/CUDA)."""
    emit(train(workspace(), name, upstream, epochs=epochs, batch_size=batch_size))


@app.command("voice-use-checkpoint")
def voice_use_checkpoint(
    name: str,
    checkpoint: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    """Select a fine-tuned checkpoint after listening review."""
    use_checkpoint(workspace(), name, checkpoint)
    emit({"voice": name, "checkpoint": str(checkpoint.resolve())})


if __name__ == "__main__":
    app()
