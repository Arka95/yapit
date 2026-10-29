from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from yapit.media import ffmpeg
from yapit.models import Brief
from yapit.pipeline import Pipeline
from yapit.scripts import load_script
from yapit.storage import Workspace
from yapit.training import train, use_checkpoint
from yapit.voices import enroll, load_voice

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
PathOption = Annotated[Path | None, typer.Option()]


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
    consent: Annotated[bool, typer.Option(help="I own this voice and consent to cloning.")] = False,
) -> None:
    """Curate local recordings with manually verified transcripts."""
    voice = enroll(workspace(), name, manifest, consent)
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
    from yapit.desktop import main

    main()


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
    voice: str,
    seconds: float = 30,
    script: PathOption = None,
    topic: str = "",
    kind: str = "travel",
    facts: str = "",
    media: PathOption = None,
    auto_script: bool = False,
) -> None:
    """Orchestrate script -> local narration -> 1080x1920 edit guide."""
    root = workspace()
    if script is not None and auto_script:
        raise typer.BadParameter("Choose --script or --auto-script, not both.")
    rate = load_voice(root, voice).words_per_minute if script or auto_script else 150
    brief = Brief(topic=topic, seconds=seconds, kind=kind, facts=facts, words_per_minute=rate)
    result = Pipeline(root).reel(
        brief,
        voice,
        load_script(script) if script else None,
        media,
        auto_script,
    )
    emit(result)


@app.command()
def narrate(
    voice: str,
    script: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    seconds: float = 30,
) -> None:
    """Render only narration MP3 and the measured clip timeline."""
    root = workspace()
    emit(
        Pipeline(root).reel(
            Brief(seconds=seconds),
            voice,
            load_script(script),
            render_video=False,
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
