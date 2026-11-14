import os
from collections import Counter
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pytest

from yapit.common_utils.metrics import record_stat, reset_stats, stats_directory

_run = {}


@pytest.fixture(autouse=True)
def isolated_source_test_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("YAPIT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("YAPIT_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("YAPIT_MODELS_DIR", str(tmp_path / "models"))


def pytest_collection_modifyitems(config, items):
    _run.clear()
    _run.update(
        id=uuid4().hex,
        started=getattr(config, "_yapit_test_started_at", perf_counter()),
        root=stats_directory(),
        outcomes={},
        mode=config.getoption("--development-mode"),
        metrics_enabled=os.environ.get("YAPIT_METRICS", "1") != "0",
    )
    reset_stats(root=_run["root"])


def pytest_runtest_logreport(report):
    if not _run:
        return
    parts = Path(report.nodeid.split("::")[0]).parts
    module = parts[1] if len(parts) > 1 and parts[0] == "src" else "root"
    if report.failed:
        _run["outcomes"][report.nodeid] = "failed"
    elif (report.when == "call" or report.skipped) and _run["outcomes"].get(
        report.nodeid
    ) != "failed":
        _run["outcomes"][report.nodeid] = report.outcome
    record_stat(
        module,
        "tests",
        {
            "run_id": _run["id"],
            "test": report.nodeid,
            "phase": report.when,
            "duration_s": report.duration,
            "outcome": report.outcome,
        },
        root=_run["root"],
        enabled=_run["metrics_enabled"],
    )


def pytest_sessionfinish(session, exitstatus):
    if not _run:
        return
    record_stat(
        "tests",
        "runs",
        {
            "run_id": _run["id"],
            "duration_s": perf_counter() - _run["started"],
            "collected": session.testscollected,
            "exit_code": int(exitstatus),
            "development_mode": _run["mode"],
            "outcomes": dict(Counter(_run["outcomes"].values())),
        },
        root=_run["root"],
        enabled=_run["metrics_enabled"],
    )
