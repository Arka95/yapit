from __future__ import annotations

import os
from pathlib import Path

from yapit.common_utils.fileutils import (
    copy_file_exclusive,
    data_filename,
    file_sha256,
    reserve_artifact_name,
    safe_name,
)
from yapit.constants import BACKGROUND_MUSIC_DIRECTORY, DEFAULT_PROFILE, GENERATED_VIDEO_DIRECTORY


class OutputArtifacts:
    def __init__(self, workspace: Workspace, source: str, profile: str):
        self.metadata = workspace.root / "outputs"
        self.audio = workspace.data.generated_audio
        self.video = workspace.data.generated_video
        self.transcripts = workspace.data.transcripts
        self.name = reserve_artifact_name(
            source,
            profile,
            [self.metadata, self.audio, self.video, self.transcripts],
            self.metadata,
            "status.json",
        )

    def path(self, suffix: str, directory: Path | None = None) -> Path:
        return (directory or self.metadata) / f"{self.name}.{suffix.lower()}"


class Workspace:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.data = DataStore(self.root.parent / "data")

    def inside(self, value: str | Path, *, exists: bool = True) -> Path:
        path = Path(value)
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(self.root) and not path.is_relative_to(self.data.root):
            raise ValueError("MCP paths must stay inside YAPIT_WORKSPACE or YAPIT_DATA_DIR.")
        if exists and not path.is_file():
            raise ValueError(f"File not found: {path}")
        return path

    def voice_path(self, name: str) -> Path:
        return self.data.voice_path(name)

    def profile_dir(self, name: str) -> Path:
        return self.data.profile_dir(name)

    def output(self, source: str, profile: str = DEFAULT_PROFILE) -> OutputArtifacts:
        return OutputArtifacts(self, source, profile)

    def voice_audio_dir(self, name: str) -> Path:
        return self.data.voice_audio_dir(name)


class DataStore:
    def __init__(self, default: Path | None = None):
        fallback = (
            default
            or Path(os.environ.get("YAPIT_WORKSPACE", "workspace")).resolve().parent / "data"
        )
        self.root = Path(os.environ.get("YAPIT_DATA_DIR") or fallback).resolve()
        self.audio = self.root / "audio"
        self.transcripts = self.root / "transcripts"
        self.generated_audio = self.root / "generated_audio"
        self.generated_video = self.root / GENERATED_VIDEO_DIRECTORY
        self.background_music = self.root / BACKGROUND_MUSIC_DIRECTORY
        for directory in (
            self.audio,
            self.transcripts,
            self.generated_audio,
            self.generated_video,
            self.background_music,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def inside(self, value: str | Path, *, exists: bool = True) -> Path:
        path = Path(value)
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Paths must stay inside YAPIT_DATA_DIR.")
        if exists and not path.is_file():
            raise ValueError(f"File not found: {path}")
        return path

    def profile_dir(self, name: str) -> Path:
        name = safe_name(name.strip()).lower()
        if name in {
            "audio",
            "transcripts",
            "generated_audio",
            GENERATED_VIDEO_DIRECTORY,
            BACKGROUND_MUSIC_DIRECTORY,
        }:
            raise ValueError("Profile name conflicts with a shared data directory.")
        return self.inside(self.root / name, exists=False)

    def voice_path(self, name: str) -> Path:
        folder = self.profile_dir(name)
        return folder / data_filename("profile", folder.name, ".json")

    def voice_audio_dir(self, name: str) -> Path:
        return self.profile_dir(name) / "audio"

    @staticmethod
    def within(directory: Path, output: Path | None = None) -> Path:
        path = (output if output is not None else directory).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError(f"Output must stay inside {directory}.")
        return path

    def archive_recording(self, source: Path, profile: str = DEFAULT_PROFILE) -> Path:
        source = source.resolve()
        if source.is_relative_to(self.audio) or source.is_relative_to(self.generated_audio):
            return source
        digest = file_sha256(source)
        target = (
            self.audio / "imports" / digest / data_filename(source.stem, profile, source.suffix)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_sha256(target) != digest:
                raise ValueError(
                    f"Archived recording was modified; refusing to overwrite {target}."
                )
            return target
        try:
            copy_file_exclusive(source, target)
        except FileExistsError:
            if file_sha256(target) != digest:
                raise ValueError(f"Recording archive collision: {target}.") from None
            return target
        return target
