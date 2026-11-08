import numpy as np
import pytest

from yapit.common_utils.mediautils import SAMPLE_RATE, decode_audio, run_media, write_wav
from yapit.editor.mixing import AudioMix, audio_filters, prepare_audio_inputs
from yapit.models import Cue, Timeline


def mix_audio(tmp_path, settings, *, with_music=True, pause=False):
    times = np.arange(4 * SAMPLE_RATE) / SAMPLE_RATE
    voice = (0.2 * np.sin(2 * np.pi * 1000 * times)).astype(np.float32)
    if pause:
        voice[SAMPLE_RATE : 2 * SAMPLE_RATE] = 0
    audio, music = tmp_path / "voice.wav", tmp_path / "music.wav"
    write_wav(audio, voice)
    write_wav(music, (0.2 * np.sin(2 * np.pi * 200 * times)).astype(np.float32))
    timeline = Timeline(
        seconds=4,
        cues=[
            Cue(start=0, end=2, text="First.", visual="Landscape", beat=0),
            Cue(start=2, end=4, text="Second.", visual="Landscape", beat=1),
        ],
    )
    inputs, seconds = prepare_audio_inputs(timeline, audio, music if with_music else None, settings)
    graph = audio_filters(
        timeline,
        settings,
        seconds=seconds,
        music=with_music,
        voice_input="0:a",
        music_input="1:a",
    )
    output = tmp_path / "mixed.wav"
    run_media(
        [
            "-y",
            "-i",
            str(audio),
            *inputs,
            "-filter_complex",
            graph,
            "-map",
            "[mixed]",
            "-t",
            str(seconds),
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )
    return decode_audio(output), seconds


def amplitude(data, frequency, start, end):
    begin, finish = round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)
    time = np.arange(begin, finish) / SAMPLE_RATE
    return 2 * abs(np.mean(data[begin:finish] * np.exp(-2j * np.pi * frequency * time)))


def test_music_fades_do_not_fade_narration_and_outro_is_reserved(tmp_path):
    audio, seconds = mix_audio(
        tmp_path,
        AudioMix(
            music_volume=30,
            music_fade_in=0.5,
            music_fade_out=0.6,
            duck_music=False,
        ),
    )
    assert amplitude(audio, 1000, 0.05, 0.15) == pytest.approx(
        amplitude(audio, 1000, 0.6, 0.8), rel=0.02
    )
    assert amplitude(audio, 200, 0.03, 0.08) < amplitude(audio, 200, 0.6, 0.8) * 0.25
    assert (
        amplitude(audio, 200, seconds - 0.08, seconds - 0.03)
        < amplitude(audio, 200, 0.6, 0.8) * 0.25
    )
    assert seconds - 4 >= 0.75
    assert amplitude(audio, 1000, 4.1, 4.5) < 0.001


def test_music_ducks_for_real_speech_and_recovers_in_pause(tmp_path):
    audio, _ = mix_audio(tmp_path, AudioMix(music_volume=50, music_fade_in=0), pause=True)
    assert amplitude(audio, 200, 0.4, 0.8) < amplitude(audio, 200, 1.5, 1.8) * 0.6
    assert amplitude(audio, 200, 2.4, 2.8) < amplitude(audio, 200, 1.5, 1.8) * 0.6


def test_beat_music_levels_modulate_only_music(tmp_path):
    audio, _ = mix_audio(
        tmp_path,
        AudioMix(
            music_volume=40,
            music_fade_in=0,
            music_fade_out=0,
            duck_music=False,
            beat_music_levels={0: 100, 1: 25},
        ),
    )
    assert amplitude(audio, 200, 2.7, 3) == pytest.approx(
        amplitude(audio, 200, 0.7, 1) * 0.25, rel=0.03
    )
    assert amplitude(audio, 1000, 2.7, 3) == pytest.approx(amplitude(audio, 1000, 0.7, 1), rel=0.02)


def test_no_music_and_muted_narration_are_silent_through_closing_hold(tmp_path):
    audio, seconds = mix_audio(tmp_path, AudioMix(audio_volume=0), with_music=False)
    assert np.max(np.abs(audio)) == 0
    assert len(audio) / SAMPLE_RATE == pytest.approx(seconds, abs=1 / SAMPLE_RATE)
