"""Opt-in stdio protocol integration tests."""

import json
import os
import sys
from pathlib import Path

import anyio
import numpy as np
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from yapit.common_utils.fileutils import save_json
from yapit.common_utils.mediautils import SAMPLE_RATE, write_wav
from yapit.models import Cue, Timeline
from yapit.storage import DataStore


def tone(seconds=5, frequency=220):
    time = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


@pytest.mark.parametrize("entry", ["module", "console"])
def test_stdio_mcp_handshake_tools_prompt_and_path_confinement(tmp_path, monkeypatch, entry):
    monkeypatch.setenv("YAPIT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("YAPIT_MODELS_DIR", str(tmp_path / "models"))
    workspace = tmp_path / "workspace"

    async def exercise():
        executable = Path(sys.executable).with_name(
            "yapit-mcp.exe" if os.name == "nt" else "yapit-mcp"
        )
        parameters = StdioServerParameters(
            command=sys.executable if entry == "module" else str(executable),
            args=["-m", "yapit.mcp"] if entry == "module" else [],
            env={**os.environ, "YAPIT_WORKSPACE": str(workspace), "YAPIT_ALLOW_CLOUD": "false"},
        )
        with anyio.fail_after(45):
            async with stdio_client(parameters) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = {tool.name for tool in (await session.list_tools()).tools}
                    assert {
                        "prepare_script_prompt",
                        "generate_script",
                        "generate_reel",
                        "generate_narration",
                        "enroll_voice",
                        "render_timeline",
                        "get_job_status",
                    } == tools
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
                    save_json(manifest, {"samples": samples})
                    enrolled = await session.call_tool(
                        "enroll_voice",
                        {"name": "mcp-data", "manifest_path": str(manifest), "consent": True},
                    )
                    assert not enrolled.isError
                    render_audio = data.generated_audio / "renderer-input.wav"
                    write_wav(render_audio, tone(5))
                    render_plan = data.transcripts / "renderer-timeline.json"
                    save_json(
                        render_plan,
                        Timeline(
                            seconds=5,
                            cues=[
                                Cue(start=0, end=5, text="Test narration.", visual="Test", beat=0)
                            ],
                        ),
                    )
                    rendered = await session.call_tool(
                        "render_timeline",
                        {"timeline_path": str(render_plan), "audio_path": str(render_audio)},
                    )
                    assert not rendered.isError
                    video = json.loads(rendered.content[0].text)
                    assert Path(video["video"]).parent == data.generated_video
                    assert Path(video["video"]).is_file()
                    assert not list((workspace / "outputs").glob("*.mp4"))
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
