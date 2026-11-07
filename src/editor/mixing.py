from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

from pydantic import Field

from yapit.common_utils.mediautils import SAMPLE_RATE, SUPPORTED_AUDIO, decode_audio
from yapit.models import Model, Timeline

Percent = Annotated[float, Field(ge=0, le=100)]
Fade = Annotated[float, Field(ge=0, le=250)]


class AudioMix(Model):
    audio_volume: Percent = 100
    music_volume: Percent = 20
    music_fade_in: Fade = 0.5
    music_fade_out: Fade = 1
    duck_music: bool = True
    loop_music: bool = True
    end_buffer: Annotated[float, Field(ge=0.5, le=10)] = 0.75
    beat_music_levels: dict[Annotated[int, Field(ge=0)], Percent] = Field(default_factory=dict)


def video_seconds(timeline: Timeline, mix: AudioMix) -> float:
    return math.ceil((timeline.seconds + mix.end_buffer) * timeline.fps) / timeline.fps


def _levels(timeline: Timeline, levels: dict[int, float]) -> str:
    if set(levels) - {cue.beat for cue in timeline.cues}:
        raise ValueError("Music dynamics reference a beat absent from the timeline.")
    current = levels.get(timeline.cues[0].beat, 100) / 100
    initial = current
    expression = f"{current:.8f}"
    for cue in timeline.cues[1:]:
        target = levels.get(cue.beat, 100) / 100
        if target != current:
            ramp = min(0.25, cue.end - cue.start)
            expression += f"+({target - current:.8f})*clip((t-{cue.start:.8f})/{ramp:.8f},0,1)"
        current = target
    return f"if(isnan(t),{initial:.8f},{expression})"


def audio_filters(
    timeline: Timeline, audio: Path, music: Path | None, mix: AudioMix
) -> tuple[list[str], str, float]:
    audio = audio.resolve()
    if not audio.is_file() or audio.suffix.lower() not in SUPPORTED_AUDIO:
        raise ValueError("Supply a supported local narration audio file.")
    actual = len(decode_audio(audio, max_seconds=max(190, timeline.seconds + 0.1))) / SAMPLE_RATE
    if abs(actual - timeline.seconds) > 0.1:
        raise ValueError("Timeline and narration duration differ by more than 100 ms.")
    seconds = (
        math.ceil((max(actual, timeline.seconds) + mix.end_buffer) * timeline.fps) / timeline.fps
    )
    voice = (
        f"[1:a]aresample=48000,aformat=channel_layouts=stereo,"
        f"volume={mix.audio_volume / 100:.8f},apad=whole_dur={seconds:.8f},"
        f"atrim=duration={seconds:.8f},asetpts=PTS-STARTPTS"
    )
    if music is None:
        return [], voice + "[mixed]", seconds
    music = music.resolve()
    if not music.is_file() or music.suffix.lower() not in SUPPORTED_AUDIO:
        raise ValueError("Supply a supported local music audio file.")
    if music.stat().st_size > 250 * 1024 * 1024:
        raise ValueError("Music input limit is 250 MB.")
    if mix.music_fade_in + mix.music_fade_out > seconds:
        raise ValueError("Music fade durations must fit the full video including its closing hold.")
    expression = _levels(timeline, mix.beat_music_levels)
    bed = (
        f"[2:a]aresample=48000,aformat=channel_layouts=stereo,"
        f"apad=whole_dur={seconds:.8f},atrim=duration={seconds:.8f},asetpts=PTS-STARTPTS,"
        f"volume='{mix.music_volume / 100:.8f}*({expression})':eval=frame"
    )
    if mix.music_fade_in:
        bed += f",afade=t=in:st=0:d={mix.music_fade_in:.8f}"
    if mix.music_fade_out:
        bed += f",afade=t=out:st={seconds - mix.music_fade_out:.8f}:d={mix.music_fade_out:.8f}"
    if mix.duck_music:
        graph = (
            voice + ",asplit=2[voice][side];" + bed + "[bed];"
            "[bed][side]sidechaincompress=threshold=0.025:ratio=8:attack=15:release=250[ducked];"
        )
    else:
        graph = voice + "[voice];" + bed + "[ducked];"
    graph += (
        "[voice][ducked]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
        "alimiter=limit=0.95:level=false:latency=true[mixed]"
    )
    extra_inputs = ["-stream_loop", "-1"] if mix.loop_music else []
    return [*extra_inputs, "-i", str(music)], graph, seconds
