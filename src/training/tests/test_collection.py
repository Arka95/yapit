import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from yapit.enrollment import load_voice
from yapit.training.dataset import load_training_samples, training_manifest
from yapit.training.desktop import MODEL_CHOICES, TrainingPanel
from yapit.training.session import TrainingSession


def test_training_reuses_legacy_clips_without_changing_the_reference(training_workspace):
    root = training_workspace
    before = root.voice_path("test-voice").read_bytes()
    session = TrainingSession(root.data, "test-voice")
    assert len(session.draft.readings) == 16
    assert len(session.samples) == 16
    assert session.ready
    session.finish()
    assert root.voice_path("test-voice").read_bytes() == before
    assert len(load_voice(root.data, "test-voice").samples) == 1
    assert len(load_training_samples(root.data, "test-voice")) == 16


def test_new_training_collection_resumes_and_preserves_exact_prompt_pairing(training_workspace):
    root = training_workspace
    training_manifest(root.data, "test-voice").unlink()
    session = TrainingSession(root.data, "test-voice")
    assert session.samples == []
    assert not session.ready
    prompt = session.draft.readings[0].text
    raw = session.new_take()
    raw.write_bytes(np.full(6000, 0.1, dtype=np.float32).tobytes())
    session.save_take(raw, True, False)
    resumed = TrainingSession(root.data, "test-voice")
    assert resumed.index == 1
    assert resumed.samples[0].text == prompt
    assert raw.exists()
    assert not training_manifest(root.data, "test-voice").exists()
    assert len(load_voice(root.data, "test-voice").samples) == 1


def test_training_dataset_portable_paths_and_minimum_do_not_replace_profile(training_workspace):
    root = training_workspace
    path = training_manifest(root.data, "test-voice")
    original = path.read_bytes()
    session = TrainingSession(root.data, "test-voice")
    session._existing = session._existing[:2]
    with pytest.raises(ValueError, match="at least 10"):
        session.finish()
    assert path.read_bytes() == original
    session = TrainingSession(root.data, "test-voice")
    session.finish()
    saved = json.loads(path.read_text())
    assert all(not Path(row["audio"]).is_absolute() for row in saved["samples"])
    assert {Path(row["audio"]).name for row in saved["samples"]} == {
        f"test-voice.audio_sample_{number}.wav" for number in range(1, 17)
    }
    if os.sep == "\\":
        path.write_text(path.read_text().replace("\\\\", "/"), encoding="utf-8")
    assert all(
        Path(sample.audio).is_file() for sample in load_training_samples(root.data, "test-voice")
    )


def test_train_button_passes_selected_model_and_publishes_only_training_data(
    training_workspace, tmp_path, monkeypatch
):
    panel = TrainingPanel.__new__(TrainingPanel)
    panel.app = SimpleNamespace(
        busy=False,
        recording=False,
        studio=SimpleNamespace(session=SimpleNamespace(workspace=training_workspace)),
    )
    panel.session, panel.recorder, panel.status, panel.work = Mock(), Mock(), Mock(), Mock()
    panel.session.alias = "test-voice"
    panel.take = None
    upstream = tmp_path / "upstream"
    (upstream / "finetuning").mkdir(parents=True)
    for filename in ("prepare_data.py", "sft_12hz.py", "dataset.py"):
        (upstream / "finetuning" / filename).touch()
    choice = list(MODEL_CHOICES)[-1]
    panel.model = SimpleNamespace(get=lambda: choice)
    panel.upstream = SimpleNamespace(get=lambda: str(upstream))
    panel.epochs = SimpleNamespace(get=lambda: "2")
    panel.batch_size = SimpleNamespace(get=lambda: "1")
    submit = Mock(return_value={"checkpoints": []})
    monkeypatch.setattr("yapit.training.desktop.run_training", submit)
    panel.start_training()
    panel.work.call_args.args[0]()
    panel.session.finish.assert_called_once_with()
    submit.assert_called_once_with(
        training_workspace,
        "test-voice",
        upstream,
        epochs=2,
        batch_size=1,
        model=MODEL_CHOICES[choice],
    )
