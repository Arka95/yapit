from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from yapit.common_utils.mediautils import ffmpeg
from yapit.constants import DEFAULT_PROFILE, DEFAULT_SPEECH_MODEL
from yapit.editor import AudioMix, generate_video
from yapit.enrollment import enroll
from yapit.enrollment.cli import data_store
from yapit.enrollment.cli import profiles as profiles_command
from yapit.enrollment.styles import STYLE_GUIDES, enroll_style, select_voice
from yapit.models import Brief, DeliveryStyle, Timeline
from yapit.musicgen.cli import music_download, music_search
from yapit.musicgen.cli import music_plan as music_plan_command
from yapit.musicgen.models import MusicPlan
from yapit.musicgen.planner import plan_music
from yapit.pipeline import Pipeline
from yapit.rescript.cli import app as rescript_app
from yapit.rescript.cli import extract_audio, transcript
from yapit.storage import Workspace
from yapit.training import resume_checkpoint, select_checkpoint, train
from yapit.yapper import generate_audio
from yapit.yapper.cli import app as yapper_app
from yapit.yapper.cli import generate as audiogen_command
from yapit.yapper.script_planner import load_script

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
PathOption = Annotated[Path | None, typer.Option()]

app.command("audiogen")(audiogen_command)
app.command("extract_audio")(extract_audio)
app.command("transcript")(transcript)
app.command("profiles")(profiles_command)
app.command("music-plan")(music_plan_command)
app.command("music-search")(music_search)
app.command("music-download")(music_download)

app.add_typer(rescript_app, name="rescript", hidden=True)
app.add_typer(yapper_app, name="yapper", hidden=True)


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


@app.command("voice-worker")
def voice_worker(voice: Annotated[str, typer.Argument()] = DEFAULT_PROFILE) -> None:
    """Keep one local speech model loaded; cache unchanged speech until explicitly stopped."""
    from yapit.yapper.worker import run_worker

    run_worker(data_store(), voice)


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
    music: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    music_plan: PathOption = None,
    audio_volume: float = 100,
    music_volume: float = 20,
    music_fade_in: float = 0.5,
    music_fade_out: float = 1,
    duck_music: bool = True,
    loop_music: bool = True,
    end_buffer: float = 0.75,
) -> None:
    """Orchestrate script -> local narration -> 1080x1920 edit guide."""
    root = workspace()
    if script is not None and auto_script:
        raise typer.BadParameter("Choose --script or --auto-script, not both.")
    rate = (
        select_voice(root.data, voice, style)[0].words_per_minute if script or auto_script else 150
    )
    brief = Brief(topic=topic, seconds=seconds, kind=kind, facts=facts, words_per_minute=rate)
    result = Pipeline(root).reel(
        brief,
        voice,
        load_script(script) if script else None,
        media,
        auto_script,
        style=style,
        source_name=media.stem if media else script.stem if script else None,
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
        music_plan=MusicPlan.model_validate_json(music_plan.read_text(encoding="utf-8"))
        if music_plan
        else None,
    )
    emit(result)


@app.command()
def videogen(
    timeline: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    audio: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    music: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    music_plan: PathOption = None,
    audio_volume: float = 100,
    music_volume: float = 20,
    music_fade_in: float = 0.5,
    music_fade_out: float = 1,
    duck_music: bool = True,
    loop_music: bool = True,
    end_buffer: float = 0.75,
) -> None:
    """Render existing narration/timeline with optional music; never synthesize or search."""
    timing = Timeline.model_validate_json(timeline.read_text(encoding="utf-8"))
    plan = None
    if music is not None:
        plan = (
            MusicPlan.model_validate_json(music_plan.read_text(encoding="utf-8"))
            if music_plan
            else plan_music(timing)
        )
    emit(
        generate_video(
            timing,
            audio,
            workspace=workspace(),
            music=music,
            mix=AudioMix(
                audio_volume=audio_volume,
                music_volume=music_volume,
                music_fade_in=music_fade_in,
                music_fade_out=music_fade_out,
                duck_music=duck_music,
                loop_music=loop_music,
                end_buffer=end_buffer,
                beat_music_levels=plan.beat_levels if plan else {},
            ),
        )
    )


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
        generate_audio(
            load_script(script),
            workspace=root,
            voice_name=voice,
            seconds=seconds,
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


@app.command("voice-use-checkpoint", hidden=True)
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
