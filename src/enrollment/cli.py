from __future__ import annotations

import json
from pathlib import Path

import typer
from dotenv import load_dotenv

from yapit.common_utils.metrics import configure_logging
from yapit.enrollment import launcher
from yapit.enrollment.profiles import profile_names
from yapit.storage import DataStore

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


def data_store() -> DataStore:
    load_dotenv()
    configure_logging()
    return DataStore(Path("data"))


@app.command("enroll-voice")
def enroll_voice() -> None:
    """Open the app to choose a profile, record your voice and finish enrollment."""
    load_dotenv()
    configure_logging()
    try:
        result = launcher.enroll_voice()
    except (RuntimeError, OSError) as exc:
        typer.echo(f"Enrollment: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, indent=2))


@app.command("profiles")
def profiles() -> None:
    """List saved profiles, including incomplete recording sessions."""
    typer.echo(json.dumps(profile_names(data_store()), indent=2))
