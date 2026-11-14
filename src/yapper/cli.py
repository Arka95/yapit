from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from yapit.common_utils.metrics import configure_logging
from yapit.constants import DEFAULT_PROFILE
from yapit.models import DeliveryStyle
from yapit.storage import Workspace
from yapit.yapper import generate_audio
from yapit.yapper.script_planner import load_script

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.callback()
def main() -> None:
    """Generate narration from supplied text or Script JSON."""


@app.command()
def generate(
    text: Annotated[str | None, typer.Option(help="Exact words to speak.")] = None,
    script: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="Text or Script JSON file.")
    ] = None,
    voice: str = DEFAULT_PROFILE,
    seconds: Annotated[
        float | None,
        typer.Option(help="Optional target (+/-33%); omit for natural duration."),
    ] = None,
    style: DeliveryStyle = DeliveryStyle.default,
    backend: Annotated[
        str | None, typer.Option(help="TTS backend or installed plugin name.")
    ] = None,
    model: Annotated[
        str | None, typer.Option(help="Model identifier/path for that backend.")
    ] = None,
) -> None:
    """Generate local narration and measured timing, without video or script rewriting."""
    if (text is None) == (script is None):
        raise typer.BadParameter("Choose exactly one of --text or --script.")
    load_dotenv()
    configure_logging()
    try:
        result = generate_audio(
            load_script(script) if script is not None else text,
            workspace=Workspace(os.environ.get("YAPIT_WORKSPACE", "workspace")),
            voice_name=voice,
            seconds=seconds,
            style=style,
            source_name=script.stem if script is not None else None,
            backend_name=backend,
            model=model,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        typer.echo(f"Yapper: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
