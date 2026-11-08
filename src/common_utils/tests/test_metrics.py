import json
import logging

import pytest

from yapit.common_utils.metrics import record_stat, reset_stats, timed


@pytest.fixture
def stats(tmp_path, monkeypatch):
    monkeypatch.setenv("YAPIT_REPO_STATS_DIR", str(tmp_path))
    monkeypatch.delenv("YAPIT_METRICS", raising=False)
    return tmp_path


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


def test_fresh_run_replaces_old_stats_and_keeps_current_measurements(stats):
    old_files = [
        stats / "editor" / "functions.jsonl",
        stats / "enrollment" / "tests.jsonl",
        stats / "tests" / "runs.jsonl",
    ]
    for path in old_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"old": true}\n')
    unrelated = stats / "editor" / "notes.txt"
    unrelated.write_text("keep")

    record_stat("editor", "functions", {"function": "first", "duration_s": 1.236})
    record_stat("editor", "functions", {"function": "second", "duration_s": 0.004})
    records = [json.loads(line) for line in old_files[0].read_text().splitlines()]
    assert [record["function"] for record in records] == ["first", "second"]
    assert [record["duration_s"] for record in records] == [1.24, 0.0]
    assert all(path.read_text() == "" for path in old_files[1:])

    reset_stats(root=stats)
    record_stat("editor", "functions", {"function": "next_run"})
    assert json.loads(old_files[0].read_text())["function"] == "next_run"
    assert unrelated.read_text() == "keep"


def test_inherited_run_keeps_parent_measurements(stats, monkeypatch):
    path = stats / "editor" / "functions.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"function": "parent"}\n')
    monkeypatch.setenv("YAPIT_REPO_STATS_ACTIVE_ROOT", str(stats.resolve()))

    record_stat("editor", "functions", {"function": "child"})

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["function"] for record in records] == ["parent", "child"]


def test_session_recording_can_remain_enabled_when_runtime_metrics_are_disabled(stats, monkeypatch):
    path = stats / "editor" / "tests.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"old": true}\n')
    monkeypatch.setenv("YAPIT_METRICS", "0")

    record_stat("editor", "tests", {"phase": "call"}, enabled=True)

    record = json.loads(path.read_text())
    assert record["phase"] == "call"
    assert "old" not in record
