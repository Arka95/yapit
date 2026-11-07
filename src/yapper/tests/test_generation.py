import json
from pathlib import Path

import pytest

from yapit.common_utils.mediautils import SAMPLE_RATE, decode_audio, write_wav
from yapit.enrollment.styles import enroll_style
from yapit.models import DeliveryStyle, Script, Timeline
from yapit.yapper import generate_audio

from .conftest import ToneSpeech, script, tone


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

    monkeypatch.setattr("yapit.yapper.generation.QwenSpeech", speech)
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
