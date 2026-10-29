import json
import os
import sys
from types import SimpleNamespace

import anyio
import numpy as np
import pytest
from conftest import tone
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import ValidationError
from typer.testing import CliRunner

from yapit.cli import app
from yapit.models import Beat, Brief, Script
from yapit.scripts import build_prompt, generate_script
from yapit.speech import QwenSpeech
from yapit.voices import load_voice


def test_cli_prompt_and_input_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("YAPIT_WORKSPACE", str(tmp_path))
    result = CliRunner().invoke(app, ["prompt", "--topic", "One weekend", "--seconds", "30"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "awaiting_script"
    assert "Target spoken words" in build_prompt(Brief(topic="Test", seconds=30))
    with pytest.raises(ValidationError):
        Brief(seconds=float("nan"))
    with pytest.raises(ValidationError):
        Script(title="Bad", beats=[Beat(text="  ")])


def test_cloud_adapter_requires_opt_in_and_sends_only_context(monkeypatch, tmp_path):
    monkeypatch.setenv("YAPIT_ALLOW_CLOUD", "false")
    with pytest.raises(ValueError, match="ALLOW_CLOUD"):
        generate_script(Brief(topic="Trip"), [])
    monkeypatch.setenv("YAPIT_ALLOW_CLOUD", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    calls = []
    expected = Script(
        title="A short reel",
        beats=[
            Beat(
                text="Plan one simple route and leave a little room to explore.",
                pause_ms=0,
            )
        ],
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


@pytest.mark.parametrize("trained", [False, True])
def test_qwen_uses_reference_or_checkpoint_without_cloud(enrolled, monkeypatch, trained):
    calls = []
    model = SimpleNamespace(
        create_voice_clone_prompt=lambda **kwargs: calls.append(("prompt", kwargs)) or ["cached"],
        generate_voice_clone=lambda **kwargs: calls.append(("clone", kwargs)) or ([tone(1)], 24000),
        generate_custom_voice=lambda **kwargs: (
            calls.append(("trained", kwargs)) or ([tone(1)], 24000)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float32="float32",
            bfloat16="bfloat16",
            float16="float16",
            cuda=SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        SimpleNamespace(
            Qwen3TTSModel=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: (
                    calls.append(("load", (args, kwargs))) or model
                ),
            ),
        ),
    )
    monkeypatch.setenv("YAPIT_DEVICE", "cpu")
    voice = load_voice(enrolled, "test-voice")
    if trained:
        voice.checkpoint = "local-checkpoint"
        voice.speaker = voice.name
    backend = QwenSpeech(voice, enrolled.voice_path(voice.name).parent)
    assert np.isfinite(backend.synthesize("First sentence.")).all()
    backend.synthesize("Second sentence.")
    assert sum(kind == "load" for kind, _ in calls) == 1
    assert sum(kind == "prompt" for kind, _ in calls) == (0 if trained else 1)
    assert sum(kind == ("trained" if trained else "clone") for kind, _ in calls) == 2


def test_stdio_mcp_handshake_tools_prompt_and_path_confinement(tmp_path):
    async def exercise():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "yapit.server"],
            env={**os.environ, "YAPIT_WORKSPACE": str(tmp_path), "YAPIT_ALLOW_CLOUD": "false"},
        )
        with anyio.fail_after(45):
            async with stdio_client(parameters) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = {tool.name for tool in (await session.list_tools()).tools}
                    assert {"generate_reel", "generate_narration", "enroll_voice"} <= tools
                    prompt = await session.get_prompt("reel_script", {"topic": "A calm weekend"})
                    assert "game changer" in prompt.messages[0].content.text
                    result = await session.call_tool(
                        "prepare_script_prompt",
                        {"topic": "A practical weekend", "seconds": 20},
                    )
                    assert not result.isError
                    response = json.loads(result.content[0].text)
                    assert response["status"] == "awaiting_script"
                    denied = await session.call_tool(
                        "prepare_script_prompt",
                        {"media_path": "..\\outside.jpg"},
                    )
                    assert denied.isError
                    missing = await session.call_tool(
                        "generate_narration",
                        {
                            "voice": "not-enrolled",
                            "seconds": 5,
                            "script": {"title": "Test", "beats": [{"text": "A short test."}]},
                        },
                    )
                    assert missing.isError

    anyio.run(exercise)
