from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from yapit.constants import MODEL_CACHE_DIRECTORY, models_directory
from yapit.enrollment.profiles import load_voice
from yapit.models import Voice
from yapit.storage import DataStore


@dataclass(frozen=True)
class LocalModel:
    label: str
    identifier: str


def _metadata(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _has_weights(directory: Path) -> bool:
    if (directory / "model.safetensors").is_file():
        return True
    weights = _metadata(directory / "model.safetensors.index.json").get("weight_map")
    return (
        isinstance(weights, dict)
        and bool(weights)
        and all(isinstance(name, str) and (directory / name).is_file() for name in weights.values())
    )


def _qwen_models(
    root: Path, caches: list[Path], voice: Voice | None, profile: Path, configured: str | None
) -> list[LocalModel]:
    candidates = [
        Path(path)
        for path in (configured, voice.checkpoint if voice else None, os.environ.get("YAPIT_MODEL"))
        if path
    ]
    candidates.extend(path.parent for path in sorted(root.glob("*/config.json")))
    for cache in caches:
        candidates.extend(
            path.parent
            for path in sorted(cache.glob("models--Qwen--Qwen3-TTS-*/snapshots/*/config.json"))
        )
    if voice:
        candidates.extend(
            path.parent for path in sorted((profile / "checkpoints").glob("*/*/config.json"))
        )
    options, seen = [], set()
    for candidate in candidates:
        directory = candidate.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        config = _metadata(directory / "config.json")
        kind = config.get("tts_model_type")
        if config.get("model_type") != "qwen3_tts" or kind not in ("base", "custom_voice"):
            continue
        if not _has_weights(directory) or not _has_weights(directory / "speech_tokenizer"):
            continue
        name = directory.name
        if directory.parent.name == "snapshots":
            name = directory.parent.parent.name.removeprefix("models--Qwen--")
        if kind == "custom_voice":
            talker = config.get("talker_config")
            speakers = talker.get("spk_id", {}) if isinstance(talker, dict) else {}
            if not isinstance(speakers, dict):
                continue
            if voice is None or (voice.speaker or voice.name) not in speakers:
                continue
            name = f"{voice.name} - {directory.parent.name} / {name} (trained)"
        else:
            name += " (reference voice)"
        if any(option.label == name for option in options):
            continue
        options.append(LocalModel(name, str(directory)))
    return options


def _pocket_models(caches: list[Path], configured: str | None) -> list[LocalModel]:
    variants = {
        path.parent.name
        for cache in caches
        for path in cache.glob(
            "models--kyutai--pocket-tts/snapshots/*/languages/*/model.safetensors"
        )
        if path.is_file()
    }
    options = [
        LocalModel(f"PocketTTS - {variant}", variant)
        for variant in sorted(variants, key=lambda variant: (variant != configured, variant))
    ]
    if configured and Path(configured).suffix.lower() in {".yaml", ".yml"}:
        path = Path(configured)
        if path.is_file():
            options.insert(
                0, LocalModel(f"PocketTTS - {path.stem} (configured)", str(path.resolve()))
            )
    return options


def local_models(store: DataStore, voice_name: str, backend: str) -> list[LocalModel]:
    """Inventory compatible local weights; never import SDKs or fetch remote assets."""
    root = models_directory()
    caches = [root / MODEL_CACHE_DIRECTORY]
    extra = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not extra and os.environ.get("HF_HOME"):
        extra = str(Path(os.environ["HF_HOME"]) / MODEL_CACHE_DIRECTORY)
    if extra and Path(extra).resolve() not in caches:
        caches.append(Path(extra).resolve())
    configured = (
        os.environ.get("YAPIT_TTS_MODEL")
        if backend == os.environ.get("YAPIT_TTS_BACKEND", "qwen")
        else None
    )
    if backend == "pocket":
        return _pocket_models(caches, configured)
    if backend != "qwen":
        if configured and Path(configured).exists():
            return [
                LocalModel(f"{backend} - {Path(configured).name}", str(Path(configured).resolve()))
            ]
        return []
    try:
        voice = load_voice(store, voice_name) if voice_name else None
    except (OSError, ValueError):
        voice = None
    profile = store.profile_dir(voice_name) if voice else store.root
    return _qwen_models(root, caches, voice, profile, configured)
