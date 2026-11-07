from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from yapit.common_utils.fileutils import save_json
from yapit.enrollment import load_voice
from yapit.storage import Workspace


def validate_checkpoint(checkpoint: Path, name: str) -> dict:
    checkpoint = checkpoint.resolve()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("tts_model_type") != "custom_voice":
        raise ValueError("Select an upstream fine-tuned CustomVoice checkpoint, not a Base model.")
    speakers = config.get("talker_config", {}).get("spk_id", {})
    if name not in speakers or not (checkpoint / "model.safetensors").is_file():
        raise ValueError("Checkpoint is incomplete or does not contain this speaker name.")
    return config


def select_checkpoint(workspace: Workspace, name: str, checkpoint: Path) -> None:
    voice = load_voice(workspace.data, name)
    checkpoint = checkpoint.resolve()
    validate_checkpoint(checkpoint, voice.name)
    voice.checkpoint = str(checkpoint)
    voice.speaker = voice.name
    save_json(workspace.voice_path(voice.name), voice)


def _architecture(config: dict) -> dict:
    result = config.copy()
    result.pop("tts_model_type", None)
    result.pop("_name_or_path", None)
    talker = result.get("talker_config", {}).copy()
    talker.pop("spk_id", None)
    talker.pop("spk_is_dialect", None)
    result["talker_config"] = talker
    return result


def _link_or_copy(source: str | Path, destination: str | Path) -> None:
    try:
        os.link(Path(source).resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_resume_model(base: Path, checkpoint: Path, destination: Path, name: str) -> None:
    """Restore Base training components while retaining fine-tuned checkpoint weights."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = validate_checkpoint(checkpoint, name)
    base_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    if (
        base_config.get("tts_model_type") != "base"
        or _architecture(config) != _architecture(base_config)
        or config["talker_config"]["spk_id"][name] != 3000
    ):
        raise ValueError("Resume checkpoint is incompatible with the configured Base trainer.")
    shapes = {}
    for shard in sorted(base.glob("model*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                if key in shapes:
                    raise ValueError("Base snapshot has duplicate tensor keys.")
                shapes[key] = (shard, tuple(tensors.get_slice(key).get_shape()))
    encoders = {key for key in shapes if key.startswith("speaker_encoder.")}
    if not encoders:
        raise ValueError("Base snapshot lacks the speaker encoder needed for continuation.")
    weights = {}
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
        if not set(shapes) - encoders <= keys or not keys <= set(shapes):
            raise ValueError("Resume checkpoint has missing or unexpected model tensors.")
        for key in keys:
            if tuple(tensors.get_slice(key).get_shape()) != shapes[key][1]:
                raise ValueError(f"Resume checkpoint tensor shape differs: {key}.")
            weights[key] = tensors.get_tensor(key)
    for key in encoders - weights.keys():
        with safe_open(str(shapes[key][0]), framework="pt", device="cpu") as tensors:
            weights[key] = tensors.get_tensor(key)
    destination.mkdir(parents=True, exist_ok=True)
    for source in base.iterdir():
        if source.name == "config.json" or (
            source.is_file()
            and (
                source.suffix == ".safetensors"
                or source.name.endswith(".safetensors.index.json")
                or source.name.startswith("pytorch_model")
            )
        ):
            continue
        target = destination / source.name
        if source.is_dir():
            shutil.copytree(source, target, copy_function=_link_or_copy)
        else:
            _link_or_copy(source, target)
    save_json(destination / "config.json", base_config)
    save_file(weights, str(destination / "model.safetensors"), metadata={"format": "pt"})
