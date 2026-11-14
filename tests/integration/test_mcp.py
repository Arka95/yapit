"""Opt-in stdio protocol integration tests."""

import os
import sys
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.parametrize("entry", ["module", "console"])
def test_stdio_mcp_handshake_tools_and_path_confinement(tmp_path, monkeypatch, entry):
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
                    denied = await session.call_tool(
                        "extract_script",
                        {"media_path": "..\\outside.wav"},
                    )
                    assert denied.isError
                    missing = await session.call_tool(
                        "generate_audio",
                        {
                            "voice": "not-enrolled",
                            "seconds": 5,
                            "script": {"title": "Test", "beats": [{"text": "A short test."}]},
                        },
                    )
                    assert missing.isError

    anyio.run(exercise)
