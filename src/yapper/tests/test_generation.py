import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from yapit.common_utils.mediautils import SAMPLE_RATE, decode_audio, write_wav
from yapit.enrollment.styles import enroll_style
from yapit.models import Beat, DeliveryStyle, Script, Timeline
from yapit.storage import Workspace
from yapit.yapper import generate_audio

from .conftest import ToneSpeech, script, tone

TEST_SAMPLE_RATE = 100


def test_plain_text_is_spoken_without_script_generation_or_video(enrolled, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Audio generation must not write scripts with an LLM or render video.")

    monkeypatch.setattr("yapit.yapper.script_planner.generate_script", forbidden)
    monkeypatch.setattr("yapit.editor.render.render_guide", forbidden)
    speech = ToneSpeech()
    result = generate_audio(
        "First sentence. Second sentence.",
        workspace=enrolled,
        voice_name="test-voice",
        seconds=7,
        backend=speech,
    )
    assert speech.calls == ["First sentence.", "Second sentence."]
    saved = Script.model_validate_json(Path(result["script"]).read_text())
    assert [beat.pause_ms for beat in saved.beats] == [180, 180]
    raw = Path(result["wav"]).with_suffix(".raw.wav")
    assert len(decode_audio(raw)) == round((6.2 + 0.36) * SAMPLE_RATE)
    timing = Timeline.model_validate_json(Path(result["timeline"]).read_text())
    assert timing.cues[-1].end == result["seconds"]
    assert result["video"] is None and result["captions"] is None
    assert not (tmp_path / "models").exists()
    assert json.loads(Path(result["provenance"]).read_text())["cloud_script"] is False


def test_narration_only_skips_video(enrolled):
    result = generate_audio(
        script(), workspace=enrolled, voice_name="test-voice", seconds=7, backend=ToneSpeech()
    )
    assert Path(result["audio"]).exists()
    assert result["video"] is None
    assert not list(enrolled.data.generated_video.iterdir())
    assert not (Path(result["job"]) / "video").exists()


def test_hard_speech_edges_are_softened_without_mutating_backend_or_padding(enrolled):
    wave = np.full(SAMPLE_RATE, 0.1, dtype=np.float32)
    wave.setflags(write=False)
    supplied = Script(
        title="Joins",
        beats=[Beat(text="First.", pause_ms=180), Beat(text="Second.", pause_ms=0)],
    )
    result = generate_audio(
        supplied,
        workspace=enrolled,
        voice_name="test-voice",
        backend=SimpleNamespace(synthesize=lambda text: wave),
    )
    raw = decode_audio(Path(result["wav"]).with_suffix(".raw.wav"))
    pause_samples = round(0.18 * SAMPLE_RATE)
    second_start = SAMPLE_RATE + pause_samples
    assert len(raw) == 2 * SAMPLE_RATE + pause_samples
    assert raw[0] == raw[SAMPLE_RATE - 1] == raw[second_start] == raw[-1] == 0
    assert np.all(raw[SAMPLE_RATE:second_start] == 0)
    assert raw[SAMPLE_RATE // 2] == pytest.approx(0.1, abs=1 / 32768)
    assert raw[second_start + SAMPLE_RATE // 2] == pytest.approx(0.1, abs=1 / 32768)
    assert np.all(wave == 0.1)


def test_duration_failure_is_persisted_without_truncating_speech(enrolled):
    with pytest.raises(ValueError, match="Rewrite"):
        generate_audio(
            script(), workspace=enrolled, voice_name="test-voice", seconds=5, backend=ToneSpeech(8)
        )
    status_files = list((enrolled.root / "outputs").glob("*.status.json"))
    assert len(status_files) == 1
    status = json.loads(status_files[0].read_text())
    assert status["status"] == "failed"
    assert not list(enrolled.data.generated_audio.glob("*.mp3"))


def test_style_generation_passes_real_reference_and_preserves_script(
    enrolled, tmp_path, monkeypatch
):
    source = tmp_path / "lively.wav"
    write_wav(source, tone(4, 260))
    reference = enroll_style(
        enrolled.data,
        "test-voice",
        DeliveryStyle.lively_vlog,
        source,
        "Look at these little faces! What a lovely surprise.",
        True,
    )
    selected = []

    def speech(voice, directory):
        selected.append((voice, directory))
        return ToneSpeech()

    monkeypatch.setattr(
        "yapit.yapper.generation.backend_for",
        lambda voice, directory, **kwargs: speech(voice, directory),
    )
    result = generate_audio(
        script(),
        workspace=enrolled,
        voice_name="test-voice",
        seconds=7,
        style=DeliveryStyle.lively_vlog,
    )
    voice, directory = selected[0]
    assert directory / voice.reference_audio == Path(reference.audio)
    assert voice.reference_text == reference.text
    assert Script.model_validate_json(Path(result["script"]).read_text()) == script()
    provenance = json.loads(Path(result["provenance"]).read_text())
    assert provenance["delivery_style"] == "lively-vlog"
    assert provenance["reference_audio"] == reference.audio
    assert Path(result["audio"]).exists()


def test_flat_named_outputs_preserve_previous_render(enrolled):
    first = generate_audio(
        script(),
        workspace=enrolled,
        voice_name="test-voice",
        seconds=7,
        backend=ToneSpeech(),
        source_name="test-voice.trip",
    )
    original = Path(first["wav"]).read_bytes()
    second = generate_audio(
        script(),
        workspace=enrolled,
        voice_name="test-voice",
        seconds=7,
        backend=ToneSpeech(3.2),
        source_name="test-voice.trip",
    )
    assert Path(first["wav"]).name == "test-voice.trip.wav"
    assert Path(second["wav"]).name == "test-voice.trip.1.wav"
    assert Path(first["wav"]).read_bytes() == original
    assert Path(second["audio"]).name == "test-voice.trip.1.mp3"
    assert all(path.is_file() for path in enrolled.data.generated_audio.iterdir())
    assert all(path.is_file() for path in (enrolled.root / "outputs").iterdir())
    assert Path(first["script"]).parent == enrolled.data.transcripts


def test_missing_style_fails_before_synthesis_or_job_creation(enrolled):
    speech = ToneSpeech()
    with pytest.raises(ValueError, match="enroll-style"):
        generate_audio(
            script(),
            workspace=enrolled,
            voice_name="test-voice",
            seconds=7,
            backend=speech,
            style=DeliveryStyle.lively_vlog,
        )
    assert not speech.calls
    assert not (enrolled.root / "outputs").exists()


@pytest.fixture
def mocked_narration(tmp_path, monkeypatch):
    root = Workspace(tmp_path / "workspace")
    voice = SimpleNamespace(
        name="test-voice", checkpoint=None, reference_audio="reference.wav", words_per_minute=150
    )
    speech = SimpleNamespace(synthesize=Mock())
    decoder = Mock(side_effect=lambda *args, **kwargs: speech.synthesize.return_value.copy())
    monkeypatch.setattr("yapit.yapper.generation.SAMPLE_RATE", TEST_SAMPLE_RATE)
    monkeypatch.setattr("yapit.yapper.timeline.SAMPLE_RATE", TEST_SAMPLE_RATE)
    monkeypatch.setattr("yapit.yapper.generation.select_voice", lambda *args: (voice, tmp_path))
    monkeypatch.setattr("yapit.yapper.generation.write_wav", Mock())
    monkeypatch.setattr("yapit.yapper.generation.encode_mp3", Mock())
    monkeypatch.setattr("yapit.yapper.generation.decode_audio", decoder)
    return root, speech, decoder


@pytest.mark.parametrize("target,duration", [(None, 300), (30, 20.1), (30, 39.9)])
def test_audio_preserves_natural_length_and_only_checks_explicit_targets(
    mocked_narration, target, duration
):
    root, speech, decoder = mocked_narration
    speech.synthesize.return_value = np.full(
        round(duration * TEST_SAMPLE_RATE), 0.1, dtype=np.float32
    )
    supplied = Script(title="Natural", beats=[Beat(text="One reviewed sentence.", pause_ms=0)])
    result = generate_audio(
        supplied,
        workspace=root,
        backend=speech,
        **({"seconds": target} if target is not None else {}),
    )
    assert result["seconds"] == pytest.approx(duration)
    assert result["tempo_factor"] == 1
    timeline = Timeline.model_validate_json(Path(result["timeline"]).read_text())
    assert timeline.cues[-1].end == pytest.approx(duration)
    assert json.loads(Path(result["brief"]).read_text())["seconds"] == target
    assert decoder.call_args.kwargs["max_seconds"] == (
        None if target is None else pytest.approx(target * 1.33)
    )


@pytest.mark.parametrize("duration", [20, 40])
def test_explicit_audio_target_rejects_outside_33_percent(mocked_narration, duration):
    root, speech, decoder = mocked_narration
    speech.synthesize.return_value = np.full(duration * TEST_SAMPLE_RATE, 0.1, dtype=np.float32)
    supplied = Script(title="Target", beats=[Beat(text="Reviewed words.", pause_ms=0)])
    with pytest.raises(ValueError, match="Rewrite"):
        generate_audio(supplied, workspace=root, backend=speech, seconds=30)
    status = json.loads(next((root.root / "outputs").glob("*.status.json")).read_text())
    assert status["status"] == "failed"
    decoder.assert_not_called()


@pytest.mark.parametrize("duration", [20, 40])
def test_encoded_audio_is_checked_again_only_for_an_explicit_target(mocked_narration, duration):
    root, speech, decoder = mocked_narration
    speech.synthesize.return_value = np.full(30 * TEST_SAMPLE_RATE, 0.1, dtype=np.float32)
    decoder.side_effect = None
    decoder.return_value = np.full(duration * TEST_SAMPLE_RATE, 0.1, dtype=np.float32)
    with pytest.raises(ValueError, match="Encoded MP3"):
        generate_audio(
            Script(title="Target", beats=[Beat(text="Reviewed words.", pause_ms=0)]),
            workspace=root,
            backend=speech,
            seconds=30,
        )


@pytest.mark.parametrize("target", [0, float("nan")])
def test_invalid_audio_target_fails_before_synthesis(mocked_narration, target):
    root, speech, _ = mocked_narration
    with pytest.raises(ValueError):
        generate_audio("Reviewed words.", workspace=root, backend=speech, seconds=target)
    speech.synthesize.assert_not_called()
