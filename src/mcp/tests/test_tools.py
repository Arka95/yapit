import json
from types import SimpleNamespace
from unittest.mock import Mock

import anyio
import pytest

from yapit.models import Beat, Script


def test_audio_tool_does_not_forward_video_mix_options(tmp_path, monkeypatch):
    from yapit.mcp.server import create_server

    monkeypatch.setenv("YAPIT_DATA_DIR", str(tmp_path / "data"))
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr("yapit.api._generate_audio", generate)
    server = create_server(tmp_path / "workspace")

    async def exercise():
        await server.call_tool(
            "generate_audio",
            {
                "script": {"title": "Test", "beats": [{"text": "A useful thought."}]},
            },
        )

    anyio.run(exercise)
    assert len(calls) == 1
    assert "music" not in calls[0] and "mix" not in calls[0]
    assert calls[0]["voice_name"] == "testuser" and calls[0]["seconds"] == 30


def test_only_requested_workflows_are_exposed(tmp_path, monkeypatch):
    from yapit.mcp.server import create_server

    monkeypatch.setenv("YAPIT_DATA_DIR", str(tmp_path / "data"))
    server = create_server(tmp_path / "workspace")

    async def exercise():
        assert {tool.name for tool in await server.list_tools()} == {
            "generate_audio",
            "generate_script",
            "extract_script",
            "generate_reel",
            "enroll_voice",
        }
        assert await server.list_prompts() == []

    anyio.run(exercise)


@pytest.mark.parametrize("seconds", [None, 15])
def test_script_tool_uses_text_contract_and_returns_script_json(tmp_path, monkeypatch, seconds):
    from yapit.mcp.server import create_server

    monkeypatch.setenv("YAPIT_DATA_DIR", str(tmp_path / "data"))
    supplied = Script(title="Trip", beats=[Beat(text="A quiet morning.")])
    generate = Mock(return_value=supplied)
    monkeypatch.setattr("yapit.mcp.server.write_script", generate)
    server = create_server(tmp_path / "workspace")

    async def exercise():
        arguments = {"prompt": "A calm weekend"}
        if seconds is not None:
            arguments["seconds"] = seconds
        result = await server.call_tool("generate_script", arguments)
        content = result[0] if isinstance(result, tuple) else result
        assert json.loads(content[0].text) == supplied.model_dump(mode="json")
        tool = next(tool for tool in await server.list_tools() if tool.name == "generate_script")
        assert set(tool.inputSchema["properties"]) == {"prompt", "seconds"}
        assert tool.inputSchema["required"] == ["prompt"]

    anyio.run(exercise)
    generate.assert_called_once_with("A calm weekend", seconds=30 if seconds is None else seconds)


def test_extraction_tool_returns_script_content_and_confines_source(tmp_path, monkeypatch):
    from yapit.mcp.server import create_server

    monkeypatch.setenv("YAPIT_DATA_DIR", str(tmp_path / "data"))
    root = tmp_path / "workspace"
    server = create_server(root)
    source = root / "speech.wav"
    source.touch()
    supplied = {"scripts": [{"content": {"title": "Trip", "beats": [{"text": "Hello."}]}}]}
    extract = Mock(return_value=supplied)
    monkeypatch.setattr("yapit.mcp.server.api.extract_script", extract)

    async def exercise():
        result = await server.call_tool("extract_script", {"media_path": str(source)})
        content = result[0] if isinstance(result, tuple) else result
        assert json.loads(content[0].text) == supplied
        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match="inside"):
            await server.call_tool("extract_script", {"media_path": str(tmp_path / "outside.wav")})

    anyio.run(exercise)
    assert extract.call_count == 1
    assert extract.call_args.args == (source,)
    assert extract.call_args.kwargs["offline"] is True
    assert extract.call_args.kwargs["data"].root == tmp_path / "data"


@pytest.mark.parametrize("source_kind", ["script", "audio_path", "prompt"])
def test_reel_tool_dispatches_the_requested_source(tmp_path, monkeypatch, source_kind):
    from yapit.mcp.server import create_server

    monkeypatch.setenv("YAPIT_DATA_DIR", str(tmp_path / "data"))
    root = tmp_path / "workspace"
    server = create_server(root)
    source = root / "speech.wav"
    source.touch()
    values = {
        "script": {"title": "Trip", "beats": [{"text": "Hello."}]},
        "audio_path": str(source),
        "prompt": "A calm weekend",
    }
    generate = Mock(return_value={"status": "complete"})
    monkeypatch.setattr("yapit.mcp.server.api.generate_reel", generate)

    async def exercise():
        await server.call_tool("generate_reel", {source_kind: values[source_kind]})

    anyio.run(exercise)
    assert generate.call_count == 1
    if source_kind == "script":
        assert generate.call_args.args[0].title == "Trip"
    else:
        assert generate.call_args.args == (None,)
        expected = source if source_kind == "audio_path" else values[source_kind]
        assert generate.call_args.kwargs[source_kind] == expected


def test_enrollment_tool_confines_manifest_and_recordings(tmp_path, monkeypatch):
    from yapit.mcp.server import create_server

    data = tmp_path / "data"
    monkeypatch.setenv("YAPIT_DATA_DIR", str(data))
    server = create_server(tmp_path / "workspace")
    manifest = data / "samples.json"
    manifest.write_text('{"samples":[]}', encoding="utf-8")
    enroll = Mock(
        return_value=SimpleNamespace(name="new-voice", samples=[object()] * 5, words_per_minute=130)
    )
    monkeypatch.setattr("yapit.api.enroll", enroll)

    async def exercise():
        await server.call_tool(
            "enroll_voice", {"name": "new-voice", "manifest_path": str(manifest)}
        )

    anyio.run(exercise)
    assert enroll.call_args.args[1:] == ("new-voice", manifest, True)
    assert enroll.call_args.args[0].root == data
    assert enroll.call_args.kwargs["confined"] is True
