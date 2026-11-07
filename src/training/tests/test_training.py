import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yapit.common_utils.fileutils import save_json
from yapit.common_utils.mediautils import write_wav
from yapit.enrollment import load_voice
from yapit.models import Sample
from yapit.storage import Workspace
from yapit.training import resume_checkpoint, select_checkpoint, train
from yapit.training.checkpoints import prepare_resume_model

from .conftest import tone


@pytest.fixture
def resume_models(tmp_path, monkeypatch):
    base = tmp_path / "snapshot"
    checkpoint = tmp_path / "resume-source"
    base.mkdir()
    checkpoint.mkdir()
    config = {
        "tts_model_type": "base",
        "talker_config": {"hidden_size": 2, "spk_id": {}, "spk_is_dialect": {}},
    }
    save_json(base / "config.json", config)
    save_json(
        checkpoint / "config.json",
        {
            "tts_model_type": "custom_voice",
            "talker_config": {
                "hidden_size": 2,
                "spk_id": {"test-voice": 3000},
                "spk_is_dialect": {"test-voice": False},
            },
        },
    )
    tensors = {
        str(base / "model-00001-of-00002.safetensors"): {
            "speaker_encoder.weight": np.array([1, 2], dtype=np.float32)
        },
        str(base / "model-00002-of-00002.safetensors"): {
            "talker.weight": np.array([3, 4], dtype=np.float32)
        },
        str(checkpoint / "model.safetensors"): {
            "talker.weight": np.array([30, 40], dtype=np.float32)
        },
    }
    for filename in tensors:
        Path(filename).write_bytes(b"synthetic weights; fake SDK reads in-memory tensors")
    (base / "model.safetensors.index.json").write_text("{}")
    tokenizer = base / "speech_tokenizer"
    tokenizer.mkdir()
    (tokenizer / "model.safetensors").write_bytes(b"preserved nested tokenizer asset")
    (base / "generation_config.json").write_text("{}")

    class TensorFile:
        def __init__(self, filename, **kwargs):
            self.values = tensors[str(filename)]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def keys(self):
            return self.values.keys()

        def get_slice(self, key):
            return SimpleNamespace(get_shape=lambda: self.values[key].shape)

        def get_tensor(self, key):
            return self.values[key].copy()

    def save_file(weights, filename, **kwargs):
        assert kwargs["metadata"] == {"format": "pt"}
        tensors[filename] = {key: value.copy() for key, value in weights.items()}
        Path(filename).write_bytes(b"merged synthetic weights")

    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=TensorFile))
    monkeypatch.setitem(sys.modules, "safetensors.torch", SimpleNamespace(save_file=save_file))
    return SimpleNamespace(base=base, checkpoint=checkpoint, tensors=tensors)


@pytest.mark.parametrize("fail_sft", [False, True])
@pytest.mark.parametrize("resume_run", [False, True])
def test_training_uses_local_snapshot_holdout_and_never_promotes_automatically(
    enrolled,
    monkeypatch,
    tmp_path,
    fail_sft,
    resume_run,
    resume_models,
):
    profile = load_voice(enrolled.data, "test-voice")
    voice_dir = enrolled.voice_audio_dir(profile.name)
    for index in range(10):
        relative = f"training-{index}.wav"
        write_wav(voice_dir / relative, tone(7, 250 + index * 10))
        profile.samples.append(Sample(audio=relative, text=f"Distinct phrase number {index}."))
    save_json(enrolled.voice_path(profile.name), profile)
    upstream = tmp_path / "upstream"
    training_dir = upstream / "finetuning"
    training_dir.mkdir(parents=True)
    for filename in ["prepare_data.py", "sft_12hz.py", "dataset.py"]:
        (training_dir / filename).write_text("# Test fixture; never executed.")
    snapshot = resume_models.base
    originals = {
        path: path.read_bytes()
        for root in (snapshot, resume_models.checkpoint)
        for path in root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr("yapit.training.trainer.platform.system", lambda: "Linux")
    monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **kwargs: str(snapshot),
        ),
    )
    original_run = subprocess.run
    calls = []

    def run(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(stdout="a" * 40 if "rev-parse" in command else "")
        if len(command) > 1 and command[1].endswith("prepare_data.py"):
            calls.append(command)
            output = Path(command[command.index("--output_jsonl") + 1])
            output.write_text("{}\n")
            return SimpleNamespace()
        if len(command) > 1 and command[1].endswith("sft_12hz.py"):
            calls.append(command)
            initialization = Path(command[command.index("--init_model_path") + 1])
            if resume_run:
                assert json.loads((initialization / "config.json").read_text()) == json.loads(
                    (snapshot / "config.json").read_text()
                )
                assert initialization != snapshot and initialization != resume_models.checkpoint
                weights = resume_models.tensors[str(initialization / "model.safetensors")]
                np.testing.assert_array_equal(weights["talker.weight"], [30, 40])
                np.testing.assert_array_equal(weights["speaker_encoder.weight"], [1, 2])
            if fail_sft:
                raise subprocess.CalledProcessError(1, command)
            output = Path(command[command.index("--output_model_path") + 1])
            checkpoint = output / "checkpoint-epoch-0"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_bytes(b"fixture")
            (checkpoint / "config.json").write_text("{}")
            return SimpleNamespace()
        return original_run(command, **kwargs)

    monkeypatch.setattr("yapit.training.trainer.subprocess.run", run)

    def execute():
        if resume_run:
            return resume_checkpoint(
                enrolled, profile.name, resume_models.checkpoint, upstream, epochs=1
            )
        return train(enrolled, profile.name, upstream, epochs=1)

    if fail_sft:
        with pytest.raises(subprocess.CalledProcessError):
            execute()
    else:
        result = execute()
        assert result["status"] == "trained_needs_listening_review"
        assert Path(result["checkpoints"][0]).exists()
    statuses = list((enrolled.root / "outputs").glob("*.status.json"))
    assert len(statuses) == 1
    status = json.loads(statuses[0].read_text())
    assert status["status"] == ("failed" if fail_sft else "trained_needs_listening_review")
    transcript_dir = enrolled.data.transcripts
    prefix = statuses[0].name.removesuffix(".status.json")
    train_rows = [
        json.loads(row)
        for row in (transcript_dir / f"{prefix}.train.jsonl").read_text().splitlines()
    ]
    heldout = [
        json.loads(row)
        for row in (transcript_dir / f"{prefix}.holdout.jsonl").read_text().splitlines()
    ]
    assert not {row["audio"] for row in train_rows} & {row["audio"] for row in heldout}
    assert len({row["ref_audio"] for row in train_rows + heldout}) == 1
    sft = calls[1]
    initialization = Path(sft[sft.index("--init_model_path") + 1])
    if resume_run:
        assert not initialization.exists()
        provenance = json.loads((enrolled.root / "outputs" / f"{prefix}.training.json").read_text())
        assert provenance["resume_checkpoint"] == str(resume_models.checkpoint)
        assert "new AdamW" in provenance["optimizer"]
        assert len(provenance["resume_weights_sha256"]) == 64
    else:
        assert initialization == snapshot
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert load_voice(enrolled.data, profile.name).checkpoint is None


def test_checkpoint_selection_checks_speaker_and_model_type(enrolled, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_json(checkpoint / "config.json", {"tts_model_type": "base"})
    with pytest.raises(ValueError, match="CustomVoice"):
        select_checkpoint(enrolled, "test-voice", checkpoint)
    save_json(
        checkpoint / "config.json",
        {
            "tts_model_type": "custom_voice",
            "talker_config": {"spk_id": {"test-voice": 3000}},
        },
    )
    (checkpoint / "model.safetensors").write_bytes(b"test placeholder, not a model")
    select_checkpoint(enrolled, "test-voice", checkpoint)
    assert load_voice(enrolled.data, "test-voice").checkpoint == str(checkpoint)


def test_resume_restores_encoder_and_preserves_nested_assets_without_stale_indexes(resume_models):
    destination = resume_models.base.parent / "initialization"
    prepare_resume_model(resume_models.base, resume_models.checkpoint, destination, "test-voice")
    assert not (destination / "model.safetensors.index.json").exists()
    assert len(list(destination.glob("model*.safetensors"))) == 1
    assert (destination / "speech_tokenizer" / "model.safetensors").read_bytes() == (
        resume_models.base / "speech_tokenizer" / "model.safetensors"
    ).read_bytes()
    assert (destination / "generation_config.json").is_file()
    weights = resume_models.tensors[str(destination / "model.safetensors")]
    np.testing.assert_array_equal(weights["speaker_encoder.weight"], [1, 2])
    np.testing.assert_array_equal(weights["talker.weight"], [30, 40])


@pytest.mark.parametrize("problem", ["speaker", "architecture", "shape", "missing", "unexpected"])
def test_resume_rejects_incompatible_checkpoint_before_creating_initialization(
    resume_models, problem
):
    checkpoint = resume_models.checkpoint
    config = json.loads((checkpoint / "config.json").read_text())
    weights = resume_models.tensors[str(checkpoint / "model.safetensors")]
    if problem == "speaker":
        config["talker_config"]["spk_id"]["test-voice"] = 42
    elif problem == "architecture":
        config["talker_config"]["hidden_size"] = 3
    elif problem == "shape":
        weights["talker.weight"] = np.ones(3)
    elif problem == "missing":
        del weights["talker.weight"]
    else:
        weights["unexpected.weight"] = np.ones(2)
    save_json(checkpoint / "config.json", config)
    destination = checkpoint.parent / "initialization"
    with pytest.raises(ValueError):
        prepare_resume_model(resume_models.base, checkpoint, destination, "test-voice")
    assert not destination.exists()


def test_training_rejects_unsupported_host_before_downloading(monkeypatch, tmp_path):
    monkeypatch.setattr("yapit.training.trainer.platform.system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="Linux"):
        train(Workspace(tmp_path), "my-voice", tmp_path)


def test_module_import_does_not_initialize_training_sdks_or_models(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import yapit.yapper; "
            "assert 'yapit.training' not in sys.modules; "
            "import yapit.training; "
            "assert callable(yapit.training.train); "
            "assert not set(sys.modules).intersection({'torch', 'flash_attn', 'huggingface_hub'})",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "models").exists()
