from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from yapit.common_utils.fileutils import save_json
from yapit.musicgen import download_music, plan_music, search_music
from yapit.storage import DataStore
from yapit.yapper.script_planner import load_script

ScriptPath = Annotated[Path, typer.Option(exists=True, dir_okay=False)]


def music_plan(
    script: ScriptPath,
    video_type: str = "travel",
    emotion: str | None = None,
    output: Path | None = None,
) -> None:
    """Prepare an editable music brief from script/beat context, without network calls."""
    plan = plan_music(load_script(script), video_type=video_type, emotion=emotion)
    if output:
        save_json(output, plan)
    typer.echo(plan.model_dump_json(indent=2))


def music_search(
    script: ScriptPath,
    video_type: str = "travel",
    emotion: str | None = None,
    limit: int = 5,
    page: int = 1,
) -> None:
    """Find and rank CC0 music; sends generic music queries, not narration."""
    load_dotenv()
    plan = plan_music(load_script(script), video_type=video_type, emotion=emotion)
    try:
        candidates = search_music(plan, limit=limit, page=page)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        json.dumps(
            {
                "music_plan": plan.model_dump(mode="json"),
                "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
                "note": (
                    "Powered by Openverse; not endorsed or certified by Openverse. "
                    "Original source license is rechecked during download; "
                    "preview before publishing."
                ),
            },
            indent=2,
        )
    )


def music_download(
    id: Annotated[str, typer.Option(help="Openverse candidate ID from music-search.")],
    script: ScriptPath,
    video_type: str = "travel",
    emotion: str | None = None,
) -> None:
    """Download one CC0 track after source verification and save its licensing evidence."""
    load_dotenv()
    plan = plan_music(load_script(script), video_type=video_type, emotion=emotion)
    try:
        result = download_music(id, plan, data=DataStore())
    except (ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2))
