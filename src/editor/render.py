from __future__ import annotations

import re
from pathlib import Path

from pydantic import Field

from yapit.common_utils.fileutils import copy_file_exclusive
from yapit.common_utils.mediautils import run_media, srt_time
from yapit.common_utils.metrics import timed
from yapit.editor.mixing import AudioMix, audio_filters, prepare_audio_inputs
from yapit.editor.templates import (
    DEFAULT_STYLE,
    PlaceholderStyle,
    create_placeholder,
    placeholder_colour,
)
from yapit.models import Model, Timeline


class RenderSettings(Model):
    video_codec: str = "libx264"
    preset: str = "fast"
    crf: int = Field(default=23, ge=0, le=51)
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"


def _prepare_slates(
    timeline: Timeline,
    output: Path,
    *,
    stem: str,
    seconds: float,
    style: PlaceholderStyle,
) -> tuple[Path, str]:
    srt = []
    concat = []
    total_frames = 0
    previous_colour = None
    alternate = False
    for index, cue in enumerate(timeline.cues, start=1):
        display = cue.model_copy(update={"end": seconds}) if index == len(timeline.cues) else cue
        colour = placeholder_colour(display.end - display.start)
        alternate = not alternate if colour == previous_colour else False
        image = create_placeholder(timeline, display, index, alternate=alternate, style=style)
        previous_colour = colour
        slate = output / f"{stem}clip-{index:03}.png"
        image.save(slate)
        end_frame = max(
            total_frames + 1,
            round((seconds if index == len(timeline.cues) else cue.end) * timeline.fps),
        )
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
    return playlist, "\n".join(srt)


def _encode_video(
    timeline: Timeline,
    audio: Path,
    playlist: Path,
    rendered: Path,
    *,
    inputs: list[str],
    graph: str,
    seconds: float,
    settings: RenderSettings,
) -> None:
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
            *inputs,
            "-filter_complex",
            graph,
            "-map",
            "0:v:0",
            "-map",
            "[mixed]",
            "-vf",
            f"fps={timeline.fps},format=yuv420p",
            "-c:v",
            settings.video_codec,
            "-preset",
            settings.preset,
            "-crf",
            str(settings.crf),
            "-c:a",
            settings.audio_codec,
            "-b:a",
            settings.audio_bitrate,
            "-t",
            f"{seconds:.8f}",
            "-movflags",
            "+faststart",
            str(rendered),
        ]
    )


@timed
def render_guide(
    timeline: Timeline,
    audio: Path,
    output: Path,
    *,
    video: Path,
    captions: Path | None = None,
    prefix: str = "",
    music: Path | None = None,
    mix: AudioMix | None = None,
    render_settings: RenderSettings | None = None,
    style: PlaceholderStyle = DEFAULT_STYLE,
    actual_seconds: float | None = None,
) -> float:
    if prefix and not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", prefix):
        raise ValueError("Video prefix must be a simple ASCII filename.")
    settings = mix or AudioMix()
    inputs, seconds = prepare_audio_inputs(
        timeline, audio, music, settings, actual_seconds=actual_seconds
    )
    graph = audio_filters(timeline, settings, seconds=seconds, music=music is not None)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}." if prefix else ""
    playlist, srt = _prepare_slates(timeline, output, stem=stem, seconds=seconds, style=style)
    if captions is None:
        from yapit.storage import DataStore

        captions = DataStore().transcripts / f"{stem}captions.srt"
    captions.parent.mkdir(parents=True, exist_ok=True)
    captions.write_text(srt, encoding="utf-8")
    rendered = output / f"{stem}guide.mp4"
    _encode_video(
        timeline,
        audio,
        playlist,
        rendered,
        inputs=inputs,
        graph=graph,
        seconds=seconds,
        settings=render_settings or RenderSettings(),
    )
    video.parent.mkdir(parents=True, exist_ok=True)
    copy_file_exclusive(rendered, video)
    rendered.unlink()
    return seconds
