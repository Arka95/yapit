from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

from yapit.models import Timeline

SAMPLE_RATE = 24000


def ffmpeg() -> str:
    return os.environ.get("YAPIT_FFMPEG") or imageio_ffmpeg.get_ffmpeg_exe()


def run_media(args: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            [ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", *args],
            check=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"FFmpeg failed: {detail}") from exc


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
    # Input bytes require a pipe, rather than a temporary file per utterance.
    result = subprocess.run(
        [
            ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
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
        capture_output=True,
        check=True,
        timeout=120,
    )
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def fit_audio(raw: Path, output: Path, target: float) -> tuple[np.ndarray, float]:
    data = decode_audio(raw)
    duration = len(data) / SAMPLE_RATE
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


def media_frames(source: Path, destination: Path) -> list[Path]:
    if source.stat().st_size > 250 * 1024 * 1024:
        raise ValueError("Media limit is 250 MB; supply a smaller local file.")
    destination.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((1280, 1280))
            output = destination / "image.jpg"
            image.save(output, quality=85)
        return [output]
    if source.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}:
        raise ValueError("Use JPEG/PNG/WebP or MP4/MOV/MKV/WebM.")
    # Six sequential snapshots at 0, 5, ... 25s. Source audio is never uploaded.
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
            str(destination / "frame-%02d.jpg"),
        ]
    )
    frames = sorted(destination.glob("frame-*.jpg"))
    if not frames:
        raise ValueError("No video frames could be extracted.")
    return frames


def srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def render_guide(timeline: Timeline, audio: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    slate_dir = output / "slates"
    slate_dir.mkdir()
    srt = []
    concat = []
    total_frames = 0
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default(size=28)
    for index, cue in enumerate(timeline.cues, start=1):
        image = Image.new("RGB", (timeline.width, timeline.height), "#111111")
        draw = ImageDraw.Draw(image)
        # ASCII labels avoid implying that a slate is a real generated visual.
        draw.text((64, 160), f"CLIP {index:02} / {len(timeline.cues):02}", font=font, fill="white")
        draw.text((64, 220), f"{cue.start:.2f}s - {cue.end:.2f}s", font=font, fill="#aaaaaa")
        draw.text((64, timeline.height - 230), "Replace with your footage", font=font, fill="white")
        slate = slate_dir / f"clip-{index:03}.png"
        image.save(slate)
        end_frame = max(total_frames + 1, round(cue.end * timeline.fps))
        frames = end_frame - total_frames
        total_frames = end_frame
        concat += [f"file 'slates/{slate.name}'", f"duration {frames / timeline.fps:.9f}"]
        srt.append(f"{index}\n{srt_time(cue.start)} --> {srt_time(cue.end)}\n{cue.text}\n")
    concat.append(f"file 'slates/{slate.name}'")
    playlist = output / "slates.txt"
    playlist.write_text("\n".join(concat) + "\n", encoding="ascii")
    (output / "captions.srt").write_text("\n".join(srt), encoding="utf-8")
    run_media(
        [
            "-y",
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            str(playlist),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            f"fps={timeline.fps},format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{timeline.seconds:.6f}",
            "-movflags",
            "+faststart",
            str(output / "guide.mp4"),
        ]
    )


def split_cut_count(seconds: float, maximum: float = 4) -> int:
    return max(1, math.ceil(seconds / maximum))
