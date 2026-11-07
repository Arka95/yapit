from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from yapit.enrollment.profiles import enroll, profile_names
from yapit.storage import Workspace

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


def workspace() -> Workspace:
    load_dotenv()
    return Workspace(os.environ.get("YAPIT_WORKSPACE", "workspace"))


@app.command("enroll-app")
def enroll_app() -> None:
    """Open the local microphone enrollment UI."""
    from yapit.enrollment.desktop import main

    main()


@app.command("voice-enroll")
def voice_enroll(
    name: str, manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)]
) -> None:
    """Create or merge a profile from at least five distinct recordings."""
    try:
        voice = enroll(workspace(), name, manifest)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps({"name": voice.name, "clips": len(voice.samples)}, indent=2))


@app.command("profiles")
def profiles() -> None:
    """List saved profiles, including incomplete recording sessions."""
    typer.echo(json.dumps(profile_names(workspace()), indent=2))
