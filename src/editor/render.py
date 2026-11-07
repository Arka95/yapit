from __future__ import annotations

import re
from pathlib import Path

from yapit.common_utils.fileutils import copy_file_exclusive
from yapit.common_utils.mediautils import run_media, srt_time
from yapit.editor.templates import create_placeholder
from yapit.models import Timeline


def render_guide(
    timeline: Timeline,
    audio: Path,
    output: Path,
    *,
    video: Path,
    captions: Path | None = None,
    prefix: str = "",
) -> None:
    if prefix and not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", prefix):
        raise ValueError("Video prefix must be a simple ASCII filename.")
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}." if prefix else ""
    srt = []
    concat = []
    total_frames = 0
    for index, cue in enumerate(timeline.cues, start=1):
        image = create_placeholder(timeline, cue, index)
        slate = output / f"{stem}clip-{index:03}.png"
        image.save(slate)
        end_frame = max(total_frames + 1, round(cue.end * timeline.fps))
        frames = end_frame - total_frames
        total_frames = end_frame
        concat += [
            f"file '{slate.name}'",
            f"option framerate {timeline.fps}",
            f"duration {frames / timeline.fps:.9f}",
        ]
        srt.append(f"{index}\n{srt_time(cue.start)} --> {srt_time(cue.end)}\n{cue.text}\n")
    concat += [f"file '{slate.name}'", f"option framerate {timeline.fps}"]
    playlist = output / f"{stem}slates.txt"
    playlist.write_text("\n".join(concat) + "\n", encoding="ascii")
    if captions is None:
        from yapit.storage import DataStore

        captions = DataStore().transcripts / f"{stem}captions.srt"
    captions.parent.mkdir(parents=True, exist_ok=True)
    captions.write_text("\n".join(srt), encoding="utf-8")
    run_media(
        [
            "-y",
            "-protocol_whitelist",
            "file",
            "-f",
            "concat",
            # Per-image framerate options require a trusted, generated concat playlist.
            "-safe",
            "0",
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
            str(output / f"{stem}guide.mp4"),
        ]
    )
    video.parent.mkdir(parents=True, exist_ok=True)
    rendered = output / f"{stem}guide.mp4"
    copy_file_exclusive(rendered, video)
    rendered.unlink()
