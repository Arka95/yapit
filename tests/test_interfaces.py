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
from yapit.media import write_wav
from yapit.models import Beat, Brief, DeliveryStyle, Script
from yapit.scripts import build_prompt, generate_script
from yapit.speech import QwenSpeech
from yapit.storage import DataStore, save_json
from yapit.voices import enroll_style, load_voice, select_voice


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


@pytest.mark.parametrize("trained,styled", [(False, False), (True, False), (False, True)])
def test_qwen_uses_reference_or_checkpoint_without_cloud(
    enrolled, monkeypatch, trained, styled, tmp_path
):
    scripts = tmp_path / "venv-bin"
    scripts.mkdir()
    (scripts / ("sox.exe" if os.name == "nt" else "sox")).write_bytes(b"test executable")
    monkeypatch.setattr(sys, "executable", str(scripts / "python.exe"))
    monkeypatch.setattr("yapit.speech.shutil.which", lambda name: None)
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
    voice_dir = enrolled.voice_audio_dir(voice.name)
    if styled:
        source = tmp_path / "lively.wav"
        write_wav(source, tone(4, 260))
        enroll_style(
            enrolled,
            voice.name,
            DeliveryStyle.lively_vlog,
            source,
            "What a lovely surprise!",
            True,
        )
        voice, voice_dir = select_voice(enrolled, voice.name, DeliveryStyle.lively_vlog)
    if trained:
        voice.checkpoint = "local-checkpoint"
        voice.speaker = voice.name
    backend = QwenSpeech(voice, voice_dir)
    assert np.isfinite(backend.synthesize("First sentence.")).all()
    backend.synthesize("Second sentence.")
    assert sum(kind == "load" for kind, _ in calls) == 1
    assert sum(kind == "prompt" for kind, _ in calls) == (0 if trained else 1)
    assert sum(kind == ("trained" if trained else "clone") for kind, _ in calls) == 2
    if styled:
        prompt = next(kwargs for kind, kwargs in calls if kind == "prompt")
        assert prompt["ref_text"] == "What a lovely surprise!"
        assert prompt["ref_audio"] == str(voice_dir / voice.reference_audio)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(scripts)


def test_style_cli_registration_requires_consent_and_forwards_style(
    enrolled, monkeypatch, tmp_path
):
    monkeypatch.setenv("YAPIT_WORKSPACE", str(enrolled.root))
    source = tmp_path / "lively.wav"
    write_wav(source, tone(4, 260))
    runner = CliRunner()
    arguments = [
        "voice-style-enroll",
        "test-voice",
        "--style",
        "lively-vlog",
        "--audio",
        str(source),
        "--text",
        "What a lovely surprise!",
    ]
    refused = runner.invoke(app, arguments)
    assert refused.exit_code != 0
    assert "own voice" in refused.output
    accepted = runner.invoke(app, [*arguments, "--consent"])
    assert accepted.exit_code == 0, accepted.output
    assert json.loads(accepted.stdout)["style"] == "lively-vlog"
    calls = []
    monkeypatch.setattr(
        "yapit.cli.Pipeline.reel",
        lambda *args, **kwargs: calls.append(kwargs) or {"status": "complete"},
    )
    script = enrolled.data.transcripts / "test-script.json"
    save_json(script, Script(title="Test", beats=[Beat(text="Look at these little faces!")]))
    result = runner.invoke(
        app,
        ["narrate", "test-voice", "--script", str(script), "--style", "lively-vlog"],
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["style"] == DeliveryStyle.lively_vlog


def test_narration_defaults_to_arka_and_preserves_source_filename(enrolled, monkeypatch):
    monkeypatch.setattr("yapit.cli.workspace", lambda: enrolled)
    calls = []
    monkeypatch.setattr(
        "yapit.cli.Pipeline.reel",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"status": "complete"},
    )
    path = enrolled.data.transcripts / "trip.json"
    save_json(path, Script(title="A different title", beats=[Beat(text="A short test.")]))
    result = CliRunner().invoke(app, ["narrate", "--script", str(path)])
    assert result.exit_code == 0, result.output
    assert calls[0][0][2] == "arka"
    assert calls[0][1]["source_name"] == "trip"


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
                    status = await session.call_tool("get_job_status", {"job_id": response["name"]})
                    assert not status.isError
                    assert json.loads(status.content[0].text)["status"] == "awaiting_script"
                    data = DataStore()
                    samples = []
                    for index in range(5):
                        sample = data.audio / f"mcp-sample-{index}.wav"
                        write_wav(sample, tone(4, 180 + 30 * index))
                        samples.append({"audio": str(sample), "text": f"Clear passage {index}."})
                    manifest = data.transcripts / "mcp-samples.json"
                    save_json(
                        manifest,
                        {"samples": samples},
                    )
                    enrolled = await session.call_tool(
                        "enroll_voice",
                        {"name": "mcp-data", "manifest_path": str(manifest), "consent": True},
                    )
                    assert not enrolled.isError
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
