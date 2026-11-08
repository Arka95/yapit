import anyio


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
