import json

import numpy as np
import pytest
from typer.testing import CliRunner

from yapit.cli import app
from yapit.common_utils.fileutils import save_json
from yapit.common_utils.mediautils import SAMPLE_RATE, write_wav
from yapit.enrollment import load_voice
from yapit.enrollment.styles import enroll_style, select_voice
from yapit.enrollment.tests.integration.conftest import tone
from yapit.models import DeliveryStyle


def test_style_registration_preserves_default_and_uses_paired_expressive_reference(
    enrolled, tmp_path
):
    original = load_voice(enrolled, "test-voice")
    source = tmp_path / "lively.wav"
    write_wav(source, tone(4, 260))
    original_bytes = source.read_bytes()
    text = "Look at these little faces! What a lovely surprise."
    reference = enroll_style(enrolled, original.name, DeliveryStyle.lively_vlog, source, text)
    persisted = load_voice(enrolled, original.name)
    assert persisted.styles[DeliveryStyle.lively_vlog] == reference
    assert reference.consent is True
    selected, directory = select_voice(enrolled, original.name, DeliveryStyle.lively_vlog)
    assert selected.reference_text == text
    assert directory / selected.reference_audio == enrolled.inside(reference.audio)
    assert directory == enrolled.root / "test-voice" / "audio"
    assert (directory / "test-voice.lively-vlog.raw.wav").read_bytes() == original_bytes
    assert source.read_bytes() == original_bytes
    default, default_dir = select_voice(enrolled, original.name)
    assert default.reference_audio == original.reference_audio
    assert default.reference_text == original.reference_text
    assert (default_dir / default.reference_audio).exists()
    transcript = enrolled.root / original.name / "test-voice.lively-vlog.reference.json"
    assert json.loads(transcript.read_text())["text"] == text
    with pytest.raises(ValueError, match="already exists"):
        enroll_style(enrolled, original.name, DeliveryStyle.lively_vlog, source, text, True)


def test_cli_style_enrollment_without_consent_flag_persists_profile(enrolled, tmp_path):
    source = tmp_path / "lively.wav"
    write_wav(source, tone(4, 260))
    text = "Look at these little faces! What a lovely surprise."
    result = CliRunner().invoke(
        app,
        [
            "enroll-style",
            "test-voice",
            "--audio",
            str(source),
            "--text",
            text,
            "--style",
            "lively-vlog",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["consent"] is True
    voice = load_voice(enrolled, "test-voice")
    selected, directory = select_voice(enrolled, voice.name, DeliveryStyle.lively_vlog)
    assert voice.styles[DeliveryStyle.lively_vlog].text == text
    assert selected.reference_text == text
    assert (directory / selected.reference_audio).is_file()


@pytest.mark.parametrize("existing_directory", [False, True])
def test_style_enrollment_requires_existing_profile_before_processing_audio(
    enrolled, tmp_path, monkeypatch, existing_directory
):
    source = tmp_path / "unprocessed.wav"
    source.touch()
    folder = enrolled.profile_dir("missing")
    if existing_directory:
        folder.mkdir()

    def unexpected_decode(*args, **kwargs):
        pytest.fail("A missing profile must be rejected before decoding audio.")

    monkeypatch.setattr("yapit.enrollment.styles.decode_audio", unexpected_decode)
    with pytest.raises(ValueError, match="Profile 'missing' does not exist"):
        enroll_style(enrolled, "missing", DeliveryStyle.lively_vlog, source, "Hello.")
    result = CliRunner().invoke(
        app, ["enroll-style", "missing", "--audio", str(source), "--text", "Hello."]
    )
    assert result.exit_code == 2, result.output
    assert "does not exist" in result.output
    assert "enroll-voice" in result.output
    assert not enrolled.voice_path("missing").exists()
    assert not (folder / "audio").exists()
    assert folder.exists() == existing_directory


def test_style_enrollment_requires_own_voice_and_usable_signal(enrolled, tmp_path):
    source = tmp_path / "silent.wav"
    write_wav(source, np.zeros(4 * SAMPLE_RATE))
    with pytest.raises(ValueError, match="own voice"):
        enroll_style(
            enrolled, "test-voice", DeliveryStyle.lively_vlog, source, "Hello.", consent=False
        )
    with pytest.raises(ValueError, match="quiet"):
        enroll_style(
            enrolled, "test-voice", DeliveryStyle.lively_vlog, source, "Hello.", consent=True
        )
    assert not load_voice(enrolled, "test-voice").styles
    assert not (enrolled.audio / "styles").exists()


def test_missing_style_and_checkpoint_cannot_silently_ignore_request(enrolled):
    with pytest.raises(ValueError, match="No 'lively-vlog' reference"):
        select_voice(enrolled, "test-voice", DeliveryStyle.lively_vlog)
    voice = load_voice(enrolled, "test-voice")
    voice.checkpoint = "selected-checkpoint"
    save_json(enrolled.voice_path(voice.name), voice)
    with pytest.raises(ValueError, match="Base voice cloning"):
        select_voice(enrolled, voice.name, DeliveryStyle.lively_vlog)
