from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from yapit.common_utils.fileutils import copy_file_exclusive
from yapit.common_utils.mediautils import run_media, srt_time
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
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}." if prefix else ""
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
        slate = output / f"{stem}clip-{index:03}.png"
        image.save(slate)
        end_frame = max(total_frames + 1, round(cue.end * timeline.fps))
        frames = end_frame - total_frames
        total_frames = end_frame
        concat += [f"file '{slate.name}'", f"duration {frames / timeline.fps:.9f}"]
        srt.append(f"{index}\n{srt_time(cue.start)} --> {srt_time(cue.end)}\n{cue.text}\n")
    concat.append(f"file '{slate.name}'")
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
            str(output / f"{stem}guide.mp4"),
        ]
    )
    video.parent.mkdir(parents=True, exist_ok=True)
    rendered = output / f"{stem}guide.mp4"
    copy_file_exclusive(rendered, video)
    rendered.unlink()
