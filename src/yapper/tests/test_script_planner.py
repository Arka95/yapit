import json
import sys
from types import SimpleNamespace

import pytest

from yapit.models import Beat, Brief, Script
from yapit.yapper.script_planner import build_prompt, generate_script, load_script


def test_prompt_contains_budget_and_schema(monkeypatch):
    monkeypatch.setenv("YAPIT_ALLOW_CLOUD", "false")
    prompt = build_prompt(Brief(topic="Test", seconds=30, words_per_minute=120))
    assert "Target spoken words: approximately 54." in prompt
    assert '"pause_ms"' in prompt
    assert '"topic": "Test"' in prompt


@pytest.mark.parametrize("extension", [".txt", ".json"])
def test_load_script_preserves_words_and_explicit_pauses(tmp_path, extension):
    supplied = Script(title="Trip", beats=[Beat(text="A quiet morning.", pause_ms=420)])
    path = tmp_path / f"trip{extension}"
    path.write_text(
        supplied.model_dump_json() if extension == ".json" else supplied.beats[0].text,
        encoding="utf-8",
    )
    parsed = load_script(path)
    assert parsed.beats[0].text == supplied.beats[0].text
    assert parsed.beats[0].pause_ms == (420 if extension == ".json" else 180)


def test_cloud_adapter_requires_opt_in_and_sends_only_context(monkeypatch, tmp_path):
    monkeypatch.setenv("YAPIT_ALLOW_CLOUD", "false")
    with pytest.raises(ValueError, match="ALLOW_CLOUD"):
        generate_script(Brief(topic="Trip"), [])
    monkeypatch.setenv("YAPIT_ALLOW_CLOUD", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    calls = []
    expected = Script(
        title="A short reel",
        beats=[Beat(text="Plan one simple route and leave a little room to explore.", pause_ms=0)],
    )

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_parsed=expected)

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            OpenAI=lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(parse=parse)),
        ),
    )
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"test-image")
    generated = generate_script(Brief(topic="Trip", seconds=5), [frame])
    assert generated == expected
    assert calls[0]["store"] is False
    assert calls[0]["input"][0]["content"][1]["type"] == "input_image"
    assert "voice" not in json.dumps(calls[0]["input"])


def test_cloud_revision_is_bounded_and_uses_duration_feedback(monkeypatch):
    monkeypatch.setenv("YAPIT_ALLOW_CLOUD", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    calls = []
    too_long = Script(title="Long", beats=[Beat(text="word " * 100, pause_ms=0)])
    short = Script(title="Short", beats=[Beat(text="One quiet morning.", pause_ms=0)])

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_parsed=too_long if len(calls) == 1 else short)

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            OpenAI=lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(parse=parse)),
        ),
    )
    assert generate_script(Brief(topic="Trip", seconds=5), []) == short
    assert len(calls) == 2
    assert "Previous draft estimated 40.0s; target 5.0s." in calls[1]["instructions"]


def test_video_planning_accepts_useful_draft_within_wider_buffer(monkeypatch):
    monkeypatch.setenv("YAPIT_ALLOW_CLOUD", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    calls = []
    supplied = Script(title="Natural", beats=[Beat(text="word " * 55, pause_ms=0)])

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_parsed=supplied)

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            OpenAI=lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(parse=parse)),
        ),
    )
    assert generate_script(Brief(seconds=30), [], duration_tolerance=9.9) == supplied
    assert len(calls) == 1
    prompt = calls[0]["input"][0]["content"][0]["text"]
    assert "Acceptable estimated duration: 20.1-39.9 seconds" in prompt
