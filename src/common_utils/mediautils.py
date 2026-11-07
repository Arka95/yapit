from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from PIL import Image

SAMPLE_RATE = 24000
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


def ffmpeg() -> str:
    return os.environ.get("YAPIT_FFMPEG") or imageio_ffmpeg.get_ffmpeg_exe()


def run_media(
    args: list[str], *, timeout: int = 600, input: bytes | None = None
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            [ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", *args],
            input=input,
            check=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"FFmpeg failed: {detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"FFmpeg timed out after {timeout} seconds.") from exc


def validate_media_source(source: Path) -> Path:
    source = source.resolve()
    if not source.is_file():
        raise ValueError(f"Media file not found: {source}")
    if source.stat().st_size > 250 * 1024 * 1024:
        raise ValueError("Media limit is 250 MB; supply a smaller local file.")
    return source


def has_video(stderr: bytes) -> bool:
    detail = stderr.decode("utf-8", errors="replace")
    return any(
        "Video:" in stream and "attached pic" not in stream
        for stream in re.findall(r"^\s*Stream #0:.*$", detail, flags=re.MULTILINE)
    )


def extract_audio_stream(
    source: Path,
    destination: Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    max_seconds: float | None = None,
) -> subprocess.CompletedProcess:
    args = ["-loglevel", "info", "-y", "-i", str(source), "-map", "0:a:0", "-vn"]
    if max_seconds is not None:
        args += ["-t", str(max_seconds + 1)]
    return run_media(
        [
            *args,
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )


def decode_audio(path: Path, *, max_seconds: float = 190) -> np.ndarray:
    result = run_media(
        [
            "-i",
            str(path.resolve()),
            "-t",
            str(max_seconds + 1),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    data = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(data) == 0 or len(data) / SAMPLE_RATE > max_seconds:
        raise ValueError(f"Audio is empty or longer than {max_seconds:g} seconds.")
    if not np.isfinite(data).all():
        raise ValueError("Audio contains invalid samples.")
    return data


def write_wav(path: Path, data: np.ndarray) -> None:
    sf.write(path, data, SAMPLE_RATE, subtype="PCM_16")


def resample(data: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == SAMPLE_RATE:
        return np.asarray(data, dtype=np.float32).reshape(-1)
    result = run_media(
        [
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ],
        input=np.asarray(data, dtype="<f4").tobytes(),
        timeout=120,
    )
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def fit_audio(
    raw: Path,
    output: Path,
    target: float,
    *,
    natural_duration_tolerance: float | None = None,
) -> tuple[np.ndarray, float]:
    if natural_duration_tolerance is not None and (
        not math.isfinite(natural_duration_tolerance) or natural_duration_tolerance < 0
    ):
        raise ValueError("Natural-duration tolerance must be finite and non-negative.")
    maximum = max(190, target + (natural_duration_tolerance or 0))
    data = decode_audio(raw, max_seconds=maximum)
    duration = len(data) / SAMPLE_RATE
    if natural_duration_tolerance is not None:
        if abs(duration - target) > natural_duration_tolerance:
            raise ValueError(
                f"Narration is {duration:.2f}s for a {target:.2f}s target, outside the "
                f"+/-{natural_duration_tolerance:.2f}s natural-duration tolerance. "
                "Rewrite the script; speech was not stretched or padded."
            )
        return data, 1.0
    factor = duration / target
    if abs(duration - target) <= 0.04:
        return data, 1.0
    if not 0.90 <= factor <= 1.10:
        if abs(duration - target) <= 5:
            return data, 1.0
        raise ValueError(
            f"Narration is {duration:.1f}s for a {target:.1f}s target. "
            "Rewrite the script or adjust the duration; refused unnatural >10% time stretching."
        )
    run_media(["-y", "-i", str(raw), "-af", f"atempo={factor:.8f}", str(output)])
    fitted = decode_audio(output)
    if abs(len(fitted) / SAMPLE_RATE - target) > 5:
        raise ValueError("Rendered narration missed the duration tolerance.")
    return fitted, factor


def encode_mp3(source: Path, destination: Path) -> None:
    run_media(
        [
            "-y",
            "-i",
            str(source),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11,alimiter=limit=0.8414:level=false",
            "-ar",
            "48000",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(destination),
        ]
    )


def media_frames(source: Path, destination: Path, *, prefix: str = "") -> list[Path]:
    source = validate_media_source(source)
    destination.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((1280, 1280))
            output = destination / f"{prefix + '.' if prefix else ''}image.jpg"
            image.save(output, quality=85)
        return [output]
    if source.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}:
        raise ValueError("Use JPEG/PNG/WebP or MP4/MOV/MKV/WebM.")
    run_media(
        [
            "-y",
            "-i",
            str(source),
            "-an",
            "-vf",
            "select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,5),"
            "scale=1280:1280:force_original_aspect_ratio=decrease",
            "-fps_mode",
            "vfr",
            "-frames:v",
            "6",
            str(destination / f"{prefix + '.' if prefix else ''}frame-%02d.jpg"),
        ]
    )
    frames = sorted(destination.glob(f"{prefix + '.' if prefix else ''}frame-*.jpg"))
    if not frames:
        raise ValueError("No video frames could be extracted.")
    return frames


def srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def split_cut_count(seconds: float, maximum: float = 4) -> int:
    return max(1, math.ceil(seconds / maximum))
