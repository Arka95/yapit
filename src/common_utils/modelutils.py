from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from yapit.constants import MODEL_CACHE_DIRECTORY, models_directory

T = TypeVar("T")


def load_model(
    loader: Callable[..., T],
    model: str | Path,
    *,
    cache_parameter: str | None = "cache_dir",
    **kwargs,
) -> T:
    cache = models_directory() / MODEL_CACHE_DIRECTORY
    if cache_parameter is not None:
        kwargs.setdefault(cache_parameter, str(cache))
    return loader(str(model), **kwargs)
