from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from yapit.common_utils.metrics import timed
from yapit.constants import MODEL_CACHE_DIRECTORY, models_directory

T = TypeVar("T")


@timed
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
    logging.getLogger(__name__).info("Loading model weights.")
    return loader(str(model), **kwargs)
