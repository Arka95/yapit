import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from yapit.constants import DEFAULT_SPEECH_MODEL
from yapit.training.jobs import run_training


def test_windows_training_requires_explicit_runtime_without_starting_processes(
    training_workspace, monkeypatch
):
    monkeypatch.setattr("yapit.training.jobs.platform.system", lambda: "Windows")
    monkeypatch.delenv("YAPIT_TRAINING_PYTHON", raising=False)
    with pytest.raises(RuntimeError, match="YAPIT_TRAINING_PYTHON"):
        run_training(
            training_workspace,
            "test-voice",
            training_workspace.root,
            epochs=3,
            batch_size=1,
            model=DEFAULT_SPEECH_MODEL,
        )


@pytest.mark.parametrize("fails", [False, True])
def test_wsl_training_maps_roots_model_and_checkpoint_receipt(
    training_workspace, monkeypatch, fails
):
    monkeypatch.setattr("yapit.training.jobs.platform.system", lambda: "Windows")
    monkeypatch.setenv("YAPIT_TRAINING_PYTHON", "/shared/bin/python")
    monkeypatch.setenv("YAPIT_TRAINING_DISTRIBUTION", "Training")
    receipt = {"status": "trained_needs_listening_review", "checkpoints": ["/data/checkpoint"]}
    execute = Mock(
        side_effect=[
            SimpleNamespace(stdout="/workspace\n/data\n/models\n/upstream\n"),
            SimpleNamespace(returncode=1 if fails else 0, stdout=json.dumps(receipt)),
            SimpleNamespace(stdout="C:\\data\\checkpoint\n"),
        ]
    )
    monkeypatch.setattr("yapit.training.jobs.subprocess.run", execute)
    arguments = dict(epochs=2, batch_size=1, model=DEFAULT_SPEECH_MODEL)
    if fails:
        with pytest.raises(RuntimeError, match="WSL training failed"):
            run_training(training_workspace, "test-voice", training_workspace.root, **arguments)
    else:
        result = run_training(
            training_workspace, "test-voice", training_workspace.root, **arguments
        )
        assert result["checkpoints"] == ["C:\\data\\checkpoint"]
    command = execute.call_args_list[1].args[0]
    assert command[:4] == ["wsl.exe", "--distribution", "Training", "--exec"]
    assert "YAPIT_DATA_DIR=/data" in command
    assert command[command.index("--model") + 1] == DEFAULT_SPEECH_MODEL
    assert command[command.index("--epochs") + 1] == "2"
    assert command[command.index("--upstream") + 1] == "/upstream"
