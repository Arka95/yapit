import json

import numpy as np
import pytest
from conftest import tone

from yapit.media import SAMPLE_RATE, write_wav
from yapit.models import DeliveryStyle
from yapit.storage import Workspace, save_json
from yapit.training import train, use_checkpoint
from yapit.voices import audio_quality, enroll, enroll_style, load_voice, select_voice


def test_enrollment_preserves_relative_samples_and_signal_report(enrolled):
    profile = load_voice(enrolled, "test-voice")
    assert profile.reference_audio in {sample.audio for sample in profile.samples}
    assert profile.consent is True
    quality = json.loads((enrolled.voice_path("test-voice").parent / "quality.json").read_text())
    assert len(quality["samples"]) == 2
    assert all(not sample["reasons"] for sample in quality["samples"])
    with pytest.raises(ValueError, match="exists"):
        enroll(
            enrolled, "test-voice", enrolled.data.transcripts / "incoming" / "samples.json", True
        )


def test_consent_duplicates_and_manifest_confinement(enrolled, tmp_path):
    manifest = enrolled.data.transcripts / "incoming" / "samples.json"
    source = enrolled.data.audio / "incoming" / "one.wav"
    with pytest.raises(ValueError, match="consent"):
        enroll(enrolled, "without-consent", manifest, False)
    save_json(
        manifest,
        {
            "samples": [
                {"audio": str(source), "text": "This is the first transcript."},
                {
                    "audio": str(source),
                    "text": "Different transcript does not make different audio.",
                },
            ]
        },
    )
    with pytest.raises(ValueError, match="Duplicate"):
        enroll(enrolled, "duplicate", manifest, True)
    assert not enrolled.voice_path("duplicate").exists()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"not audio")
    save_json(manifest, {"samples": [{"audio": str(outside), "text": "An outside path."}]})
    with pytest.raises(ValueError, match="inside"):
        enroll(enrolled, "outside", manifest, True, confined=True)
    with pytest.raises(ValueError, match="Name"):
        enrolled.voice_path("..\\escape")


@pytest.mark.parametrize(
    "data",
    [
        np.zeros(SAMPLE_RATE * 4, dtype=np.float32),
        np.ones(SAMPLE_RATE * 4, dtype=np.float32),
    ],
)
def test_signal_checks_reject_silence_and_clipping(data):
    assert audio_quality(data)["reasons"]


def test_checkpoint_selection_checks_speaker_and_model_type(enrolled, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_json(checkpoint / "config.json", {"tts_model_type": "base"})
    with pytest.raises(ValueError, match="CustomVoice"):
        use_checkpoint(enrolled, "test-voice", checkpoint)
    save_json(
        checkpoint / "config.json",
        {
            "tts_model_type": "custom_voice",
            "talker_config": {"spk_id": {"test-voice": 3000}},
        },
    )
    (checkpoint / "model.safetensors").write_bytes(b"test placeholder, not a model")
    use_checkpoint(enrolled, "test-voice", checkpoint)
    assert load_voice(enrolled, "test-voice").checkpoint == str(checkpoint)


def test_training_rejects_unsupported_host_before_downloading(monkeypatch, tmp_path):
    monkeypatch.setattr("yapit.training.platform.system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="Linux"):
        train(Workspace(tmp_path), "my-voice", tmp_path)


def test_style_registration_preserves_default_and_uses_paired_expressive_reference(
    enrolled, tmp_path
):
    original = load_voice(enrolled, "test-voice")
    source = tmp_path / "lively.wav"
    write_wav(source, tone(4, 260))
    original_bytes = source.read_bytes()
    text = "Look at these little faces! What a lovely surprise."
    reference = enroll_style(
        enrolled, original.name, DeliveryStyle.lively_vlog, source, text, consent=True
    )
    selected, directory = select_voice(enrolled, original.name, DeliveryStyle.lively_vlog)
    assert selected.reference_text == text
    assert directory / selected.reference_audio == enrolled.inside(reference.audio)
    assert directory.is_relative_to(enrolled.data.audio)
    assert (directory / "raw.wav").read_bytes() == original_bytes
    assert source.read_bytes() == original_bytes
    default, default_dir = select_voice(enrolled, original.name)
    assert default.reference_audio == original.reference_audio
    assert default.reference_text == original.reference_text
    assert (default_dir / default.reference_audio).exists()
    transcript = (
        enrolled.data.transcripts / "styles" / original.name / "lively-vlog" / "reference.json"
    )
    assert json.loads(transcript.read_text())["text"] == text
    with pytest.raises(ValueError, match="already exists"):
        enroll_style(enrolled, original.name, DeliveryStyle.lively_vlog, source, text, True)


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
    assert not (enrolled.data.audio / "styles").exists()


def test_missing_style_and_checkpoint_cannot_silently_ignore_request(enrolled):
    with pytest.raises(ValueError, match="No 'lively-vlog' reference"):
        select_voice(enrolled, "test-voice", DeliveryStyle.lively_vlog)
    voice = load_voice(enrolled, "test-voice")
    voice.checkpoint = "selected-checkpoint"
    save_json(enrolled.voice_path(voice.name), voice)
    with pytest.raises(ValueError, match="Base voice cloning"):
        select_voice(enrolled, voice.name, DeliveryStyle.lively_vlog)
