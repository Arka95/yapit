from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from yapit.constants import DEFAULT_ASR_MODEL
from yapit.rescript.audio import extract_audio as extract_audio_file
from yapit.rescript.transcription import transcribe

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.command("extract_audio")
def extract_audio(
    media: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[
        Path | None, typer.Option(help="Audio directory; defaults to data\\audio.")
    ] = None,
) -> None:
    """Extract/convert media audio or copy an already supported audio file."""
    load_dotenv()
    try:
        result = extract_audio_file(media, output_dir=output_dir)
    except (ValueError, RuntimeError, OSError) as exc:
        typer.echo(f"Rescript: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command("transcript")
def transcript(
    media: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[
        Path | None, typer.Option(help="Flat transcript directory; must be data\\transcripts.")
    ] = None,
    model: str = DEFAULT_ASR_MODEL,
    device: Annotated[
        str | None, typer.Option(help="ASR device; defaults to selected mode or cpu.")
    ] = None,
    compute_type: Annotated[
        str | None, typer.Option(help="ASR precision; defaults to selected mode or int8.")
    ] = None,
    language: str = "en",
    offline: Annotated[
        bool, typer.Option(help="Use cached/local models; --no-offline permits model downloads.")
    ] = True,
) -> None:
    """Transcribe local audio/video into reviewed-before-use Script JSON."""
    load_dotenv()
    try:
        result = transcribe(
            media,
            output=output,
            model=model,
            device=device,
            compute_type=compute_type,
            language=language,
            offline=offline,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        typer.echo(f"Rescript: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
