import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import tone

from yapit.media import write_wav
from yapit.models import Sample
from yapit.storage import save_json
from yapit.training import train
from yapit.voices import load_voice


@pytest.mark.parametrize("fail_sft", [False, True])
def test_training_uses_local_snapshot_holdout_and_never_promotes_automatically(
    enrolled,
    monkeypatch,
    tmp_path,
    fail_sft,
):
    profile = load_voice(enrolled, "test-voice")
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
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setattr("yapit.training.platform.system", lambda: "Linux")
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
            if fail_sft:
                raise subprocess.CalledProcessError(1, command)
            output = Path(command[command.index("--output_model_path") + 1])
            checkpoint = output / "checkpoint-epoch-0"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_bytes(b"fixture")
            (checkpoint / "config.json").write_text("{}")
            return SimpleNamespace()
        return original_run(command, **kwargs)

    monkeypatch.setattr("yapit.training.subprocess.run", run)
    if fail_sft:
        with pytest.raises(subprocess.CalledProcessError):
            train(enrolled, profile.name, upstream, epochs=1)
    else:
        result = train(enrolled, profile.name, upstream, epochs=1)
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
    assert sft[sft.index("--init_model_path") + 1] == str(snapshot)
    assert load_voice(enrolled, profile.name).checkpoint is None
