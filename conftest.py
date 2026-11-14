from pathlib import Path
from time import perf_counter

import pytest

ROOT = Path(__file__).parent


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
        if parts[0] == "tests" or "integration" in parts or "enrolled" in item.fixturenames:
            item.add_marker(pytest.mark.integration)
        if item.get_closest_marker("agent") and rapid:
            item.add_marker(pytest.mark.skip(reason="Agent checks are disabled in rapid mode."))
        for marker, option in (("integration", "--run-integration"), ("llm", "--run-llm")):
            if item.get_closest_marker(marker) and (rapid or not config.getoption(option)):
                item.add_marker(
                    pytest.mark.skip(reason=f"Opt in with --development-mode=full {option}.")
                )


def pytest_ignore_collect(collection_path, config):
    try:
        parts = collection_path.relative_to(ROOT).parts
    except ValueError:
        return None
    if not parts:
        return None
    rapid = config.getoption("--development-mode") == "rapid"
    if parts[0] == "agenttests":
        return rapid
    if parts[0] == "tests" or "integration" in parts:
        return rapid or not config.getoption("--run-integration")
    return None


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
    session.config._yapit_test_started_at = perf_counter()
