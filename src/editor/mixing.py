from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import soundfile as sf
from pydantic import Field

from yapit.common_utils.mediautils import SAMPLE_RATE, SUPPORTED_AUDIO, decode_audio
from yapit.common_utils.metrics import timed
from yapit.models import Model, Timeline

Percent = Annotated[float, Field(ge=0, le=100)]
Fade = Annotated[float, Field(ge=0, le=250)]
MUSIC_INPUT_LIMIT_BYTES = 250 * 1024 * 1024


class AudioMix(Model):
    audio_volume: Percent = 100
    music_volume: Percent = 20
    music_fade_in: Fade = 0.5
    music_fade_out: Fade = 1
    duck_music: bool = True
    duck_threshold: Annotated[float, Field(gt=0, le=1)] = 0.025
    duck_ratio: Annotated[float, Field(ge=1, le=20)] = 8
    duck_attack_ms: Annotated[float, Field(gt=0)] = 15
    duck_release_ms: Annotated[float, Field(gt=0)] = 250
    loop_music: bool = True
    end_buffer: Annotated[float, Field(ge=0.5, le=10)] = 0.75
    duration_tolerance: Annotated[float, Field(ge=0)] = 0.1
    beat_transition: Annotated[float, Field(gt=0)] = 0.25
    beat_music_levels: dict[Annotated[int, Field(ge=0)], Percent] = Field(default_factory=dict)


def video_seconds(timeline: Timeline, mix: AudioMix, actual_seconds: float | None = None) -> float:
    duration = max(timeline.seconds, actual_seconds or timeline.seconds)
    return math.ceil((duration + mix.end_buffer) * timeline.fps) / timeline.fps


def _levels(timeline: Timeline, levels: dict[int, float], transition: float) -> str:
    if set(levels) - {cue.beat for cue in timeline.cues}:
        raise ValueError("Music dynamics reference a beat absent from the timeline.")
    current = levels.get(timeline.cues[0].beat, 100) / 100
    initial = current
    expression = f"{current:.8f}"
    for cue in timeline.cues[1:]:
        target = levels.get(cue.beat, 100) / 100
        if target != current:
            ramp = min(transition, cue.end - cue.start)
            expression += f"+({target - current:.8f})*clip((t-{cue.start:.8f})/{ramp:.8f},0,1)"
        current = target
    return f"if(isnan(t),{initial:.8f},{expression})"


def narration_seconds(audio: Path, maximum: float) -> float:
    try:
        info = sf.info(audio)
        seconds = info.frames / info.samplerate
    except (sf.LibsndfileError, OSError):
        seconds = len(decode_audio(audio, max_seconds=maximum)) / SAMPLE_RATE
    if not math.isfinite(seconds) or not 0 < seconds <= maximum:
        raise ValueError(f"Audio is empty or longer than {maximum:g} seconds.")
    return seconds


@timed
def prepare_audio_inputs(
    timeline: Timeline,
    audio: Path,
    music: Path | None,
    mix: AudioMix,
    *,
    actual_seconds: float | None = None,
) -> tuple[list[str], float]:
    audio = audio.resolve()
    if not audio.is_file() or audio.suffix.lower() not in SUPPORTED_AUDIO:
        raise ValueError("Supply a supported local narration audio file.")
    actual = actual_seconds
    if actual is None:
        actual = narration_seconds(audio, max(190, timeline.seconds + mix.duration_tolerance))
    if not math.isfinite(actual) or actual <= 0:
        raise ValueError("Narration duration must be finite and positive.")
    if abs(actual - timeline.seconds) > mix.duration_tolerance:
        raise ValueError(
            f"Timeline and narration duration differ by more than {mix.duration_tolerance:g}s."
        )
    seconds = video_seconds(timeline, mix, actual)
    if music is None:
        return [], seconds
    music = music.resolve()
    if not music.is_file() or music.suffix.lower() not in SUPPORTED_AUDIO:
        raise ValueError("Supply a supported local music audio file.")
    if music.stat().st_size > MUSIC_INPUT_LIMIT_BYTES:
        raise ValueError("Music input limit is 250 MB.")
    if mix.music_fade_in + mix.music_fade_out > seconds:
        raise ValueError("Music fade durations must fit the full video including its closing hold.")
    extra_inputs = ["-stream_loop", "-1"] if mix.loop_music else []
    return [*extra_inputs, "-i", str(music)], seconds


def audio_filters(
    timeline: Timeline,
    mix: AudioMix,
    *,
    seconds: float,
    music: bool = False,
    voice_input: str = "1:a",
    music_input: str = "2:a",
) -> str:
    voice = (
        f"[{voice_input}]aresample=48000,aformat=channel_layouts=stereo,"
        f"volume={mix.audio_volume / 100:.8f},apad=whole_dur={seconds:.8f},"
        f"atrim=duration={seconds:.8f},asetpts=PTS-STARTPTS"
    )
    if not music:
        return voice + "[mixed]"
    expression = _levels(timeline, mix.beat_music_levels, mix.beat_transition)
    bed = (
        f"[{music_input}]aresample=48000,aformat=channel_layouts=stereo,"
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
            f"[bed][side]sidechaincompress=threshold={mix.duck_threshold:g}:"
            f"ratio={mix.duck_ratio:g}:attack={mix.duck_attack_ms:g}:"
            f"release={mix.duck_release_ms:g}[ducked];"
        )
    else:
        graph = voice + "[voice];" + bed + "[ducked];"
    graph += (
        "[voice][ducked]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
        "alimiter=limit=0.95:level=false:latency=true[mixed]"
    )
    return graph
