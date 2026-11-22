from pathlib import Path

from yapit.common_utils.fileutils import file_sha256, save_json
from yapit.enrollment import load_voice
from yapit.models import Sample
from yapit.training.dataset import (
    load_training_samples,
    migrate_profile_recordings,
    training_manifest,
)


def test_legacy_migration_moves_dataset_preserving_reference_checkpoint_and_recording_bytes(
    training_workspace,
):
    root = training_workspace
    voice = load_voice(root.data, "test-voice")
    samples = load_training_samples(root.data, voice.name)
    original = {Path(sample.audio).name: file_sha256(Path(sample.audio)) for sample in samples}
    voice.samples = [Sample(audio=Path(sample.audio).name, text=sample.text) for sample in samples]
    voice.reference_audio, voice.reference_text = voice.samples[0].audio, voice.samples[0].text
    voice.checkpoint, voice.speaker = "keep-checkpoint", voice.name
    save_json(root.voice_path(voice.name), voice)
    before = root.voice_path(voice.name).read_bytes()
    training_manifest(root.data, voice.name).unlink()
    result = migrate_profile_recordings(root.data, voice.name)
    saved = load_voice(root.data, voice.name)
    dataset = load_training_samples(root.data, voice.name)
    assert result["moved_files"] == 15
    assert len(saved.samples) == 1
    assert saved.checkpoint == "keep-checkpoint"
    assert saved.speaker == voice.name
    assert (root.voice_audio_dir(voice.name) / saved.reference_audio).is_file()
    assert {
        Path(sample.audio).name: file_sha256(Path(sample.audio)) for sample in dataset
    } == original
    archive = training_manifest(root.data, voice.name).parent
    assert (archive / "test-voice.legacy.profile.json").read_bytes() == before
    assert len(list(root.voice_audio_dir(voice.name).glob("sample-*.wav"))) == 1
    assert migrate_profile_recordings(root.data, voice.name)["status"] == "already_separated"
