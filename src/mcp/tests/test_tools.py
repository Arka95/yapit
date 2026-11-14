import json
from unittest.mock import Mock

import anyio
import pytest

from yapit.models import Beat, Script


def test_narration_tool_does_not_forward_video_mix_options(tmp_path, monkeypatch):
    from yapit.mcp.server import create_server

    monkeypatch.setenv("YAPIT_DATA_DIR", str(tmp_path / "data"))
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr("yapit.mcp.server.generate_audio", generate)
    server = create_server(tmp_path / "workspace")

    async def exercise():
        await server.call_tool(
            "generate_narration",
            {
                "script": {"title": "Test", "beats": [{"text": "A useful thought."}]},
            },
        )

    anyio.run(exercise)
    assert len(calls) == 1
    assert "music" not in calls[0] and "mix" not in calls[0]


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
