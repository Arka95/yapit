from __future__ import annotations

import json
import logging
import os
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")
_write_lock = threading.RLock()
_active_roots: set[Path] = set()


def stats_directory() -> Path:
    if override := os.environ.get("YAPIT_REPO_STATS_DIR"):
        return Path(override).resolve()
    checkout = Path(__file__).resolve().parents[2]
    root = checkout if (checkout / "pyproject.toml").is_file() else Path.cwd()
    return root / ".repo-stats"


def reset_stats(*, root: Path | None = None, enabled: bool | None = None) -> None:
    if enabled is None:
        enabled = os.environ.get("YAPIT_METRICS", "1") != "0"
    if not enabled:
        return
    directory = (root or stats_directory()).resolve()
    with _write_lock:
        try:
            for category in ("functions", "tests", "runs"):
                for path in directory.glob(f"*/{category}.jsonl"):
                    path.write_text("", encoding="utf-8")
        except OSError:
            logging.getLogger(__name__).debug(
                "Could not reset repository statistics.", exc_info=True
            )
        _active_roots.add(directory)
        os.environ["YAPIT_REPO_STATS_ACTIVE_ROOT"] = str(directory)


def record_stat(
    module: str,
    category: str,
    value: dict,
    *,
    root: Path | None = None,
    enabled: bool | None = None,
) -> None:
    if enabled is None:
        enabled = os.environ.get("YAPIT_METRICS", "1") != "0"
    if not enabled:
        return
    try:
        directory = (root or stats_directory()).resolve()
        path = directory / module / f"{category}.jsonl"
        if "duration_s" in value:
            value = {**value, "duration_s": round(value["duration_s"], 2)}
        row = json.dumps({"timestamp": datetime.now(UTC).isoformat(), **value}) + "\n"
        with _write_lock:
            if directory not in _active_roots:
                if os.environ.get("YAPIT_REPO_STATS_ACTIVE_ROOT") == str(directory):
                    _active_roots.add(directory)
                else:
                    reset_stats(root=directory, enabled=True)
            os.environ["YAPIT_REPO_STATS_ACTIVE_ROOT"] = str(directory)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as output:
                output.write(row)
    except (OSError, TypeError, ValueError):
        logging.getLogger(__name__).debug("Could not write repository statistics.", exc_info=True)


def configure_logging() -> None:
    level = os.environ.get("YAPIT_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=logging.getLevelNamesMapping().get(level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def timed(function: Callable[P, T]) -> Callable[P, T]:
    logger = logging.getLogger(function.__module__)
    parts = function.__module__.split(".")
    module = parts[1] if parts[0] == "yapit" and len(parts) > 1 else parts[0]
    name = f"{function.__module__}.{function.__qualname__}"

    @wraps(function)
    def measured(*args: P.args, **kwargs: P.kwargs) -> T:
        started = perf_counter()
        error = None
        logger.debug("%s started", name)
        try:
            return function(*args, **kwargs)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            duration = perf_counter() - started
            logger.log(
                logging.WARNING if error else logging.INFO,
                "%s %s in %.2fs",
                name,
                f"failed ({error})" if error else "completed",
                duration,
            )
            record_stat(
                module,
                "functions",
                {
                    "function": name,
                    "duration_s": duration,
                    "outcome": "failed" if error else "passed",
                    "error_type": error,
                    "context": "test" if "PYTEST_CURRENT_TEST" in os.environ else "runtime",
                },
            )

    return measured
