from __future__ import annotations

import importlib
import importlib.metadata
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from yapit.common_utils.mediautils import SAMPLE_RATE, resample
from yapit.models import Voice

MAX_CHUNK_SECONDS = 45


class SpeechBackend(Protocol):
    def synthesize(self, text: str) -> np.ndarray: ...


class ModelBackend(SpeechBackend, Protocol):
    def load(self) -> None: ...
    def close(self) -> None: ...
    def cache_identity(self) -> dict: ...
    def describe(self) -> dict: ...


@dataclass(frozen=True)
class BackendSelection:
    backend: str = "qwen"
    model: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("Supply a non-empty TTS backend name.")
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("Supply a non-empty model identifier.")


BUILTINS = {
    "qwen": "yapit.yapper.speech:QwenSpeech",
    "pocket": "yapit.yapper.pocket:PocketSpeech",
    "cosyvoice": "yapit.yapper.cosyvoice:CosyVoiceSpeech",
}
FACTORIES: dict[str, Callable[..., ModelBackend]] = {}


def register_backend(name: str, factory: Callable[..., ModelBackend]) -> None:
    """Register a trusted local adapter without changing the narration workflow."""
    if not name.strip() or not callable(factory):
        raise ValueError("Supply a backend name and callable factory.")
    FACTORIES[name] = factory


def select_backend(
    backend: str | None = None,
    model: str | None = None,
    *,
    default: BackendSelection | None = None,
) -> BackendSelection:
    default = default or BackendSelection(
        os.environ.get("YAPIT_TTS_BACKEND", "qwen"), os.environ.get("YAPIT_TTS_MODEL")
    )
    name = backend or default.backend
    return BackendSelection(
        backend=name,
        model=model if model is not None else default.model if name == default.backend else None,
    )


def create_backend(voice: Voice, directory: Path, selection: BackendSelection) -> ModelBackend:
    name = selection.backend
    factory = FACTORIES.get(name)
    if factory is None and name in BUILTINS:
        module, attribute = BUILTINS[name].split(":")
        factory = getattr(importlib.import_module(module), attribute)
    if factory is None:
        entries = importlib.metadata.entry_points(group="yapit.tts")
        entry = next((entry for entry in entries if entry.name == name), None)
        if entry is not None:
            factory = entry.load()
    if factory is None:
        raise ValueError(
            f"Unknown TTS backend {name!r}; install its yapit.tts plugin or register a factory."
        )
    kwargs = {"model": selection.model} if selection.model is not None else {}
    backend = factory(voice, directory, **kwargs)
    for method in ("load", "close", "synthesize", "cache_identity", "describe"):
        if not callable(getattr(backend, method, None)):
            raise TypeError(f"TTS backend {name!r} must implement {method}().")
    return backend


def package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def asset_stamps(paths: Iterable[Path]) -> list[tuple[str, int, int]]:
    stamps = []
    for path in paths:
        path = path.resolve()
        if path.is_file():
            stat = path.stat()
            stamps.append((str(path), stat.st_size, stat.st_mtime_ns))
    return stamps


def normalize_audio(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    data = np.asarray(wave, dtype=np.float32).reshape(-1)
    if not len(data) or not np.isfinite(data).all() or np.max(np.abs(data)) < 1e-5:
        raise ValueError("Speech model produced empty, invalid or silent audio.")
    data = resample(data, sample_rate)
    if len(data) / SAMPLE_RATE > MAX_CHUNK_SECONDS:
        raise ValueError("A beat exceeded 45 seconds; split the script into shorter beats.")
    return data
