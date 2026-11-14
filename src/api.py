"""Canonical customer workflows shared by Python, CLI and MCP adapters."""

from __future__ import annotations

import os
from pathlib import Path

from yapit.constants import DEFAULT_PROFILE
from yapit.enrollment import enroll
from yapit.models import DeliveryStyle, Script
from yapit.pipeline import generate_reel
from yapit.rescript import extract_script
from yapit.storage import Workspace
from yapit.yapper import generate_audio as _generate_audio
from yapit.yapper import generate_script

__all__ = [
    "enroll_voice",
    "extract_script",
    "generate_audio",
    "generate_reel",
    "generate_script",
]


def generate_audio(
    script: Script | str,
    voice: str = DEFAULT_PROFILE,
    seconds: float = 30,
    *,
    workspace: Workspace | None = None,
    style: DeliveryStyle = DeliveryStyle.default,
) -> dict:
    """Speak supplied words without rewriting them or generating video."""
    return _generate_audio(
        script, voice_name=voice, seconds=seconds, workspace=workspace, style=style
    )


def enroll_voice(
    name: str,
    manifest_path: Path,
    consent: bool = True,
    *,
    workspace: Workspace | None = None,
    confined: bool = False,
) -> dict:
    """Import verified recordings; MCP callers confine manifest and recordings to data."""
    root = workspace or Workspace(os.environ.get("YAPIT_WORKSPACE", "workspace"))
    manifest = root.data.inside(manifest_path) if confined else Path(manifest_path)
    voice = enroll(root.data, name, manifest, consent, confined=confined)
    return {
        "name": voice.name,
        "clips": len(voice.samples),
        "words_per_minute": voice.words_per_minute,
    }
