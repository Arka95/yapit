import os
from collections import Counter
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pytest

from yapit.common_utils.metrics import record_stat, reset_stats, stats_directory

ROOT = Path(__file__).parent
_run = {}


def pytest_addoption(parser):
    parser.addoption(
        "--development-mode",
        choices=("rapid", "full"),
        default="rapid",
        help="Rapid runs unit tests only; full allows explicitly selected agent/integration tests.",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        help="Enable integration tests only when explicitly requested.",
    )
    parser.addoption(
        "--run-llm", action="store_true", help="Enable model tests only when explicitly requested."
    )


def pytest_collection_modifyitems(config, items):
    rapid = config.getoption("--development-mode") == "rapid"
    for item in items:
        parts = item.path.relative_to(ROOT).parts
        if "agenttests" in parts:
            item.add_marker(pytest.mark.agent)
        if "integration" in parts or "enrolled" in item.fixturenames:
            item.add_marker(pytest.mark.integration)
        if item.get_closest_marker("agent") and rapid:
            item.add_marker(pytest.mark.skip(reason="Agent checks are disabled in rapid mode."))
        for marker, option in (("integration", "--run-integration"), ("llm", "--run-llm")):
            if item.get_closest_marker(marker) and (rapid or not config.getoption(option)):
                item.add_marker(
                    pytest.mark.skip(reason=f"Opt in with --development-mode=full {option}.")
                )


@pytest.fixture(autouse=True)
def unit_tests_do_not_launch_processes(request, monkeypatch):
    if any(request.node.get_closest_marker(name) for name in ("integration", "llm", "agent")):
        return

    def forbidden(*args, **kwargs):
        pytest.fail("Unit tests must mock subprocesses; mark real external work as integration.")

    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)


def pytest_sessionstart(session):
    _run.clear()
    _run.update(
        id=uuid4().hex,
        started=perf_counter(),
        root=stats_directory(),
        outcomes={},
        mode=session.config.getoption("--development-mode"),
        metrics_enabled=os.environ.get("YAPIT_METRICS", "1") != "0",
    )
    reset_stats(root=_run["root"])


def pytest_runtest_logreport(report):
    parts = Path(report.nodeid.split("::")[0]).parts
    if len(parts) > 1 and parts[0] == "src":
        module = parts[1]
    else:
        module = "agenttests" if parts[0] == "agenttests" else "root"
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
