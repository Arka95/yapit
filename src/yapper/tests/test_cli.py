import json

import pytest
from typer.testing import CliRunner

from yapit.cli import app
from yapit.common_utils.fileutils import save_json
from yapit.models import Beat, Script
from yapit.storage import Workspace
from yapit.yapper.cli import app as yapper_app


@pytest.mark.parametrize("kind", ["text", "txt", "json"])
def test_yapper_cli_accepts_text_and_script_files(tmp_path, monkeypatch, kind):
    workspace = Workspace(tmp_path / "workspace")
    monkeypatch.setenv("YAPIT_WORKSPACE", str(workspace.root))
    calls = []
    monkeypatch.setattr(
        "yapit.yapper.cli.generate_audio",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"status": "complete"},
    )
    content = "Speak these supplied words."
    if kind == "text":
        arguments = ["--text", content]
    else:
        path = workspace.data.transcripts / f"source.{kind}"
        if kind == "json":
            save_json(path, Script(title="Source", beats=[Beat(text=content)]))
        else:
            path.write_text(content)
        arguments = ["--script", str(path)]
    result = CliRunner().invoke(
        app,
        ["generate-audio", *arguments, "--voice", "test-voice", "--seconds", "7"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "complete"
    supplied = calls[0][0][0]
    assert supplied == content if kind == "text" else supplied.beats[0].text == content
    assert calls[0][1]["voice_name"] == "test-voice"
    assert calls[0][1]["source_name"] == (None if kind == "text" else "source")


@pytest.mark.parametrize("arguments", [[], ["--text", "Words.", "--script"]])
@pytest.mark.parametrize("entry", ["flat", "standalone"])
def test_yapper_cli_requires_exactly_one_input(tmp_path, arguments, entry):
    if arguments:
        path = tmp_path / "script.txt"
        path.write_text("Words.")
        arguments = [*arguments, str(path)]
    result = (
        CliRunner().invoke(app, ["generate-audio", *arguments])
        if entry == "flat"
        else CliRunner().invoke(yapper_app, ["generate", *arguments])
    )
    assert result.exit_code != 0
    assert "Choose exactly one" in result.output


def test_narration_forwards_selected_delivery_style(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")
    monkeypatch.setenv("YAPIT_WORKSPACE", str(workspace.root))
    calls = []
    monkeypatch.setattr(
        "yapit.yapper.cli.generate_audio",
        lambda *args, **kwargs: calls.append(kwargs) or {"status": "complete"},
    )
    path = workspace.data.transcripts / "script.json"
    save_json(path, Script(title="Test", beats=[Beat(text="These are approved words.")]))
    result = CliRunner().invoke(
        app,
        [
            "generate-audio",
            "--voice",
            "test-voice",
            "--script",
            str(path),
            "--style",
            "lively-vlog",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["style"].value == "lively-vlog"


def test_narration_defaults_to_testuser_and_preserves_source_filename(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")
    monkeypatch.setenv("YAPIT_WORKSPACE", str(workspace.root))
    calls = []
    monkeypatch.setattr(
        "yapit.yapper.cli.generate_audio",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"status": "complete"},
    )
    path = workspace.data.transcripts / "trip.json"
    save_json(path, Script(title="A different title", beats=[Beat(text="A short test.")]))
    result = CliRunner().invoke(app, ["generate-audio", "--script", str(path)])
    assert result.exit_code == 0, result.output
    assert calls[0][1]["voice_name"] == "testuser"
    assert calls[0][1]["source_name"] == "trip"
