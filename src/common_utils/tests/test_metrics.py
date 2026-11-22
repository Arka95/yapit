import json
import logging
import os
from pathlib import Path

import pytest

from yapit.common_utils.metrics import record_stat, reset_stats, timed


@pytest.fixture
def stats(tmp_path, monkeypatch):
    monkeypatch.setenv("YAPIT_REPO_STATS_DIR", str(tmp_path))
    monkeypatch.delenv("YAPIT_METRICS", raising=False)
    reset_stats(root=tmp_path)
    return Path(os.environ["YAPIT_REPO_STATS_ACTIVE_ROOT"])


def test_timing_preserves_result_and_records_module_duration(stats, caplog):
    def add(left, right):
        return left + right

    add.__module__ = "yapit.editor.example"
    measured = timed(add)
    with caplog.at_level(logging.INFO):
        assert measured(2, 3) == 5
    record = json.loads((stats / "editor" / "functions.jsonl").read_text())
    assert record["duration_s"] >= 0
    assert record["outcome"] == "passed"
    assert record["function"].endswith(".add")
    assert "completed" in caplog.text


def test_exception_is_preserved_without_logging_sensitive_contents(stats, caplog):
    failure = ValueError("private exception contents")

    @timed
    def fail(secret):
        raise failure

    with pytest.raises(ValueError) as caught:
        fail("private argument")
    assert caught.value is failure
    record = json.loads(next(stats.rglob("functions.jsonl")).read_text())
    assert record["outcome"] == "failed"
    assert record["error_type"] == "ValueError"
    assert "private" not in json.dumps(record) + caplog.text


def test_statistics_write_failure_does_not_change_application_result(tmp_path, monkeypatch):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("blocked")
    monkeypatch.setenv("YAPIT_REPO_STATS_DIR", str(blocked))

    @timed
    def operation():
        return "result"

    assert operation() == "result"


def test_metrics_can_be_disabled(stats, monkeypatch):
    monkeypatch.setenv("YAPIT_METRICS", "0")

    @timed
    def operation():
        return 7

    assert operation() == 7
    assert not list(stats.rglob("*.jsonl"))


def test_fresh_run_isolates_measurements_without_erasing_other_runs(stats):
    old_files = [
        stats.parent / "editor" / "functions.jsonl",
        stats.parent / "enrollment" / "tests.jsonl",
        stats.parent / "tests" / "runs.jsonl",
    ]
    for path in old_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"old": true}\n')
    unrelated = stats / "editor" / "notes.txt"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("keep")

    record_stat("editor", "functions", {"function": "first", "duration_s": 1.236})
    record_stat("editor", "functions", {"function": "second", "duration_s": 0.004})
    path = stats / "editor" / "functions.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["function"] for record in records] == ["first", "second"]
    assert [record["duration_s"] for record in records] == [1.24, 0.0]
    assert all(path.read_text() == '{"old": true}\n' for path in old_files)

    reset_stats(root=stats.parent)
    record_stat("editor", "functions", {"function": "next_run"})
    current = Path(os.environ["YAPIT_REPO_STATS_ACTIVE_ROOT"])
    assert current != stats
    assert (
        json.loads((current / "editor" / "functions.jsonl").read_text())["function"] == "next_run"
    )
    assert [json.loads(line) for line in path.read_text().splitlines()] == records
    assert unrelated.read_text() == "keep"


def test_inherited_run_keeps_parent_measurements(stats, monkeypatch):
    path = stats / "editor" / "functions.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"function": "parent"}\n')
    monkeypatch.setenv("YAPIT_REPO_STATS_ACTIVE_ROOT", str(stats.resolve()))
    monkeypatch.setattr("yapit.common_utils.metrics._active_roots", {})

    record_stat("editor", "functions", {"function": "child"})

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["function"] for record in records] == ["parent", "child"]


def test_independent_process_does_not_truncate_running_workers_stats(stats, monkeypatch):
    record_stat("yapper", "functions", {"function": "worker"})
    parent = stats / "yapper" / "functions.jsonl"
    with monkeypatch.context() as fresh:
        fresh.setattr("yapit.common_utils.metrics._active_roots", {})
        fresh.delenv("YAPIT_REPO_STATS_ACTIVE_ROOT", raising=False)
        record_stat("yapper", "functions", {"function": "independent"})
        other = Path(os.environ["YAPIT_REPO_STATS_ACTIVE_ROOT"])
        assert other != stats
        assert (
            json.loads((other / "yapper" / "functions.jsonl").read_text())["function"]
            == "independent"
        )
    assert json.loads(parent.read_text())["function"] == "worker"
    record_stat("yapper", "functions", {"function": "worker_again"})
    assert [json.loads(line)["function"] for line in parent.read_text().splitlines()] == [
        "worker",
        "worker_again",
    ]


def test_session_recording_can_remain_enabled_when_runtime_metrics_are_disabled(stats, monkeypatch):
    path = stats / "editor" / "tests.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"old": true}\n')
    monkeypatch.setenv("YAPIT_METRICS", "0")

    record_stat("editor", "tests", {"phase": "call"}, enabled=True)

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0] == {"old": True}
    assert records[1]["phase"] == "call"
    assert len(records) == 2
