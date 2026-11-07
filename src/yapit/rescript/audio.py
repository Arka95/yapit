from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import soundfile as sf

from yapit.media import SAMPLE_RATE, run_media
from yapit.storage import DataStore

# These audio containers can be passed straight to YapIt's FFmpeg decoder.
SUPPORTED_AUDIO = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".aif",
    ".aiff",
    ".wma",
}


def has_video(stderr: bytes) -> bool:
    detail = stderr.decode("utf-8", errors="replace")
    return any(
        "Video:" in stream and "attached pic" not in stream
        for stream in re.findall(r"^\s*Stream #0:.*$", detail, flags=re.MULTILINE)
    )


def extract_audio(media: Path, *, output_dir: Path | None = None) -> dict:
    source = media.resolve()
    if not source.is_file():
        raise ValueError(f"Media file not found: {source}")
    if source.stat().st_size > 250 * 1024 * 1024:
        raise ValueError("Media limit is 250 MB; supply a smaller local file.")
    folder = DataStore.within(DataStore().audio, output_dir)
    with tempfile.TemporaryDirectory(prefix="yapit-extract-") as temporary:
        decoded = Path(temporary) / "audio.wav"
        try:
            result = run_media(
                [
                    "-loglevel",
                    "info",
                    "-y",
                    "-i",
                    str(source),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-c:a",
                    "pcm_s16le",
                    str(decoded),
                ]
            )
        except RuntimeError as exc:
            raise ValueError(
                f"Cannot extract audio: unsupported/corrupt media or no audio stream. {exc}"
            ) from exc
        if sf.info(decoded).frames == 0:
            raise ValueError("Cannot extract audio: the audio stream is empty.")
        video = has_video(result.stderr)
        copy_original = not video and source.suffix.lower() in SUPPORTED_AUDIO
        name = source.name if copy_original else f"{source.stem}.wav"
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
            target = destination.open("xb")
        except FileExistsError as exc:
            raise ValueError(f"Audio output already exists: {destination}.") from exc
        with target:
            try:
                with (source if copy_original else decoded).open("rb") as incoming:
                    shutil.copyfileobj(incoming, target)
            except OSError:
                target.close()
                destination.unlink(missing_ok=True)
                raise
    return {
        "status": "complete",
        "audio": str(destination),
        "action": "copied" if copy_original else "converted",
        "format": destination.suffix.lower().lstrip("."),
    }
