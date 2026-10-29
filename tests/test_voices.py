import json

import numpy as np
import pytest

from yapit.media import SAMPLE_RATE
from yapit.storage import Workspace, save_json
from yapit.training import train, use_checkpoint
from yapit.voices import audio_quality, enroll, load_voice


def test_enrollment_preserves_relative_samples_and_signal_report(enrolled):
    profile = load_voice(enrolled, "test-voice")
    assert profile.reference_audio in {sample.audio for sample in profile.samples}
    assert profile.consent is True
    quality = json.loads((enrolled.voice_path("test-voice").parent / "quality.json").read_text())
    assert len(quality["samples"]) == 2
    assert all(not sample["reasons"] for sample in quality["samples"])
    with pytest.raises(ValueError, match="exists"):
        enroll(enrolled, "test-voice", enrolled.root / "incoming" / "samples.json", True)


def test_consent_duplicates_and_manifest_confinement(enrolled, tmp_path):
    manifest = enrolled.root / "incoming" / "samples.json"
    with pytest.raises(ValueError, match="consent"):
        enroll(enrolled, "without-consent", manifest, False)
    save_json(
        manifest,
        {
            "samples": [
                {"audio": "one.wav", "text": "This is the first transcript."},
                {"audio": "one.wav", "text": "Different transcript does not make different audio."},
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
