import json

import pytest

from yapit.models import Voice
from yapit.storage import DataStore
from yapit.yapper.model_catalog import local_models


@pytest.fixture(autouse=True)
def isolated_model_settings(monkeypatch):
    for name in (
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HOME",
        "YAPIT_MODEL",
        "YAPIT_TTS_MODEL",
        "YAPIT_TTS_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)


def qwen_model(directory, *, speaker=None):
    directory.mkdir(parents=True)
    config = {
        "model_type": "qwen3_tts",
        "tts_model_type": "custom_voice" if speaker else "base",
        "talker_config": {"spk_id": {speaker: 0}} if speaker else {},
    }
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "model.safetensors").touch()
    tokenizer = directory / "speech_tokenizer"
    tokenizer.mkdir()
    (tokenizer / "model.safetensors").touch()
    return directory.resolve()


def profile(store, checkpoint):
    voice = Voice(
        name="myvoice",
        consent=True,
        reference_audio="reference.wav",
        reference_text="Hello.",
        samples=[],
        words_per_minute=140,
        checkpoint=str(checkpoint),
    )
    store.profile_dir(voice.name).mkdir(parents=True, exist_ok=True)
    store.voice_path(voice.name).write_text(voice.model_dump_json())


def test_qwen_inventory_prioritizes_selected_checkpoint_and_filters_incompatible_models(tmp_path):
    root = tmp_path / "models"
    base = qwen_model(root / "Qwen3-TTS-Base")
    incomplete = qwen_model(root / "incomplete")
    (incomplete / "speech_tokenizer" / "model.safetensors").unlink()
    store = DataStore()
    selected = qwen_model(
        store.profile_dir("myvoice") / "checkpoints" / "run" / "epoch-2", speaker="myvoice"
    )
    qwen_model(
        store.profile_dir("myvoice") / "checkpoints" / "run" / "wrong-voice", speaker="other"
    )
    profile(store, selected)
    options = local_models(store, "myvoice", "qwen")
    assert [option.identifier for option in options] == [str(selected), str(base)]
    assert "trained" in options[0].label
    assert all(str(tmp_path) not in option.label for option in options)


def test_pocket_inventory_uses_only_cached_clone_capable_variants(tmp_path, monkeypatch):
    root = tmp_path / "models" / "hub"
    for repository, variant in (
        ("pocket-tts", "english_2026-10"),
        ("pocket-tts-without-voice-cloning", "english_2026-09"),
    ):
        folder = (
            root
            / f"models--kyutai--{repository}"
            / "snapshots"
            / "revision"
            / "languages"
            / variant
        )
        folder.mkdir(parents=True)
        (folder / "model.safetensors").touch()
    monkeypatch.setenv("YAPIT_TTS_BACKEND", "qwen")
    monkeypatch.setenv("YAPIT_TTS_MODEL", "not-a-pocket-model")
    options = local_models(DataStore(), "", "pocket")
    assert [option.identifier for option in options] == ["english_2026-10"]
    assert local_models(DataStore(), "", "unknown") == []


def test_configured_cache_and_local_models_are_both_inventoried(tmp_path, monkeypatch):
    local = qwen_model(tmp_path / "models" / "Qwen-local")
    cache = tmp_path / "shared-cache"
    cached = qwen_model(cache / "models--Qwen--Qwen3-TTS-Cached" / "snapshots" / "revision")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    options = local_models(DataStore(), "", "qwen")
    assert {option.identifier for option in options} == {str(local), str(cached)}
    assert all("revision" not in option.label for option in options)


def test_missing_configured_remote_id_does_not_create_a_download_option(tmp_path, monkeypatch):
    monkeypatch.setenv("YAPIT_TTS_BACKEND", "qwen")
    monkeypatch.setenv("YAPIT_TTS_MODEL", "Qwen/not-cached")
    assert local_models(DataStore(), "", "qwen") == []


def test_incomplete_sharded_weights_are_excluded(tmp_path):
    model = qwen_model(tmp_path / "models" / "sharded")
    (model / "model.safetensors").unlink()
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "shard-1.safetensors", "b": "shard-2.safetensors"}})
    )
    (model / "shard-1.safetensors").touch()
    assert local_models(DataStore(), "", "qwen") == []
    (model / "shard-2.safetensors").touch()
    assert [option.identifier for option in local_models(DataStore(), "", "qwen")] == [str(model)]
