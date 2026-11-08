import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from yapit.editor.mixing import AudioMix, audio_filters, prepare_audio_inputs
from yapit.models import Cue, Timeline


@pytest.mark.parametrize(
    "field,value",
    [
        ("audio_volume", -1),
        ("music_volume", 101),
        ("music_volume", float("nan")),
        ("music_fade_in", -1),
        ("end_buffer", 0.49),
    ],
)
def test_invalid_mix_controls_are_rejected(field, value):
    with pytest.raises(ValidationError):
        AudioMix(**{field: value})


@pytest.fixture
def timeline():
    return Timeline(
        seconds=4,
        cues=[
            Cue(start=0, end=2, text="First", visual="Footage", beat=0),
            Cue(start=2, end=4, text="Second", visual="Footage", beat=1),
        ],
    )


def test_known_duration_avoids_probe_and_decode(tmp_path, monkeypatch, timeline):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"unchanged narration")
    probe = Mock(side_effect=AssertionError("Unnecessary duration measurement"))
    monkeypatch.setattr("yapit.editor.mixing.narration_seconds", probe)
    inputs, seconds = prepare_audio_inputs(timeline, audio, None, AudioMix(), actual_seconds=4)
    assert inputs == []
    assert seconds == pytest.approx(4.766666667)
    assert audio.read_bytes() == b"unchanged narration"


def test_metadata_duration_does_not_decode_audio(tmp_path, monkeypatch, timeline):
    audio = tmp_path / "voice.wav"
    audio.touch()
    monkeypatch.setattr(
        "yapit.editor.mixing.sf.info", lambda path: SimpleNamespace(frames=96000, samplerate=24000)
    )
    monkeypatch.setattr(
        "yapit.editor.mixing.decode_audio", Mock(side_effect=AssertionError("Unexpected decode"))
    )
    _, seconds = prepare_audio_inputs(timeline, audio, None, AudioMix())
    assert seconds == pytest.approx(4.766666667)


def test_unsupported_metadata_format_uses_decoder_fallback(tmp_path, monkeypatch, timeline):
    audio = tmp_path / "voice.m4a"
    audio.touch()
    monkeypatch.setattr("yapit.editor.mixing.sf.info", Mock(side_effect=OSError("Unsupported")))
    decoder = Mock(return_value=bytes(96000))
    monkeypatch.setattr("yapit.editor.mixing.decode_audio", decoder)
    _, seconds = prepare_audio_inputs(timeline, audio, None, AudioMix())
    assert seconds == pytest.approx(4.766666667)
    decoder.assert_called_once_with(audio.resolve(), max_seconds=190)


def test_duration_tolerance_can_be_adjusted(tmp_path, timeline):
    audio = tmp_path / "voice.wav"
    audio.touch()
    with pytest.raises(ValueError, match="duration differ"):
        prepare_audio_inputs(timeline, audio, None, AudioMix(), actual_seconds=4.2)
    _, seconds = prepare_audio_inputs(
        timeline, audio, None, AudioMix(duration_tolerance=0.3), actual_seconds=4.2
    )
    assert seconds == pytest.approx(4.966666667)


def test_graph_uses_custom_inputs_and_ducking_settings(timeline):
    graph = audio_filters(
        timeline,
        AudioMix(duck_threshold=0.04, duck_ratio=4, duck_attack_ms=20, duck_release_ms=300),
        seconds=4.75,
        music=True,
        voice_input="0:a",
        music_input="1:a",
    )
    assert "[0:a]aresample" in graph
    assert "[1:a]aresample" in graph
    assert "threshold=0.04:ratio=4:attack=20:release=300" in graph


def test_beat_transition_is_configurable_and_unknown_beats_are_rejected(timeline):
    graph = audio_filters(
        timeline,
        AudioMix(beat_transition=0.5, beat_music_levels={1: 25}),
        seconds=4.75,
        music=True,
    )
    ramp = re.search(r"clip\(\(t-([\d.]+)\)/([\d.]+),0,1\)", graph)
    assert ramp is not None
    assert tuple(map(float, ramp.groups())) == (2, 0.5)
    with pytest.raises(ValueError, match="absent"):
        audio_filters(timeline, AudioMix(beat_music_levels={2: 25}), seconds=4.75, music=True)
