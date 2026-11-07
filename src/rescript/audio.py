from __future__ import annotations

import tempfile
from pathlib import Path

import soundfile as sf

from yapit.common_utils.fileutils import copy_file_exclusive, data_filename
from yapit.common_utils.mediautils import (
    SUPPORTED_AUDIO,
    extract_audio_stream,
    has_video,
    validate_media_source,
)
from yapit.constants import DEFAULT_PROFILE
from yapit.storage import DataStore


def extract_audio(
    media: Path, *, output_dir: Path | None = None, profile: str = DEFAULT_PROFILE
) -> dict:
    source = validate_media_source(media)
    folder = DataStore.within(DataStore().audio, output_dir)
    with tempfile.TemporaryDirectory(prefix="yapit-extract-") as temporary:
        decoded = Path(temporary) / "audio.wav"
        try:
            result = extract_audio_stream(source, decoded)
        except RuntimeError as exc:
            raise ValueError(
                f"Cannot extract audio: unsupported/corrupt media or no audio stream. {exc}"
            ) from exc
        if sf.info(decoded).frames == 0:
            raise ValueError("Cannot extract audio: the audio stream is empty.")
        video = has_video(result.stderr)
        copy_original = not video and source.suffix.lower() in SUPPORTED_AUDIO
        name = data_filename(source.stem, profile, source.suffix if copy_original else ".wav")
        destination = folder / name
        if copy_original and destination == source:
            return {
                "status": "complete",
                "audio": str(source),
                "action": "already_stored",
                "format": source.suffix.lower().lstrip("."),
            }
        if destination.exists():
            raise ValueError(f"Audio output already exists: {destination}; choose --output-dir.")
        folder.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents overwrites by simultaneous extractions.
        try:
            copy_file_exclusive(source if copy_original else decoded, destination)
        except FileExistsError as exc:
            raise ValueError(f"Audio output already exists: {destination}.") from exc
    return {
        "status": "complete",
        "audio": str(destination),
        "action": "copied" if copy_original else "converted",
        "format": destination.suffix.lower().lstrip("."),
    }
