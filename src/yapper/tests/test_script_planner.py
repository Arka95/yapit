import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from yapit.cli import app
from yapit.models import Beat, Brief, Script
from yapit.yapper import generate_script
from yapit.yapper.script_planner import _write_script, build_prompt, load_script


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


def test_cloud_adapter_requires_opt_in_and_sends_only_text(monkeypatch):
    monkeypatch.setenv("YAPIT_ALLOW_CLOUD", "false")
    with pytest.raises(ValueError, match="ALLOW_CLOUD"):
        generate_script("Trip")
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
    generated = generate_script("Trip", seconds=5)
    assert generated == expected
    assert calls[0]["store"] is False
    content = calls[0]["input"][0]["content"]
    assert len(content) == 1 and content[0]["type"] == "input_text"
    assert '"topic": "Trip"' in content[0]["text"]
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
    assert generate_script("Trip", seconds=5) == short
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
    assert _write_script(Brief(seconds=30), duration_tolerance=9.9) == supplied
    assert len(calls) == 1
    prompt = calls[0]["input"][0]["content"][0]["text"]
    assert "Acceptable estimated duration: 20.1-39.9 seconds" in prompt


@pytest.mark.parametrize(
    "prompt,seconds",
    [("", 30), ("   ", 30), ("Trip", 4), ("Trip", 181), ("Trip", float("nan"))],
)
def test_invalid_generation_request_never_constructs_cloud_client(monkeypatch, prompt, seconds):
    client = Mock()
    monkeypatch.setenv("YAPIT_ALLOW_CLOUD", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client))
    with pytest.raises(ValueError):
        generate_script(prompt, seconds)
    client.assert_not_called()


@pytest.mark.parametrize("seconds", [None, 15])
def test_generate_script_cli_uses_shared_contract_and_emits_script_json(monkeypatch, seconds):
    supplied = Script(title="Trip", beats=[Beat(text="A quiet morning.")])
    generate = Mock(return_value=supplied)
    monkeypatch.setattr("yapit.cli.write_script", generate)
    monkeypatch.setattr("yapit.cli.load_dotenv", lambda: None)
    monkeypatch.setattr("yapit.cli.configure_logging", lambda: None)
    arguments = ["generate-script", "A calm weekend"]
    if seconds is not None:
        arguments += ["--seconds", str(seconds)]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == supplied.model_dump(mode="json")
    generate.assert_called_once_with("A calm weekend", seconds=30 if seconds is None else seconds)
