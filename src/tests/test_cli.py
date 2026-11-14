import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from yapit.cli import app
from yapit.models import Beat, Brief, Script
from yapit.storage import DataStore, Workspace


def test_enrollment_cli_launches_the_owned_desktop_module(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "yapit.enrollment.cli.launcher.enroll_voice",
        lambda: calls.append(True) or {"status": "launched", "pid": 42},
    )
    result = CliRunner().invoke(app, ["enroll-voice"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert json.loads(result.stdout) == {"status": "launched", "pid": 42}


def test_training_cli_distinguishes_fresh_resume_and_inference_selection(tmp_path, monkeypatch):
    root = Workspace(tmp_path / "workspace")
    calls = []

    def operation(kind):
        def run(*args, **kwargs):
            calls.append((kind, args, kwargs))
            return {"status": "test"}

        return run

    monkeypatch.setattr("yapit.cli.train", operation("fresh"))
    monkeypatch.setattr("yapit.cli.resume_checkpoint", operation("resume"))
    monkeypatch.setattr("yapit.cli.select_checkpoint", operation("select"))
    directory = str(root.root)
    runner = CliRunner()
    training = ["voice-train", "test-voice", "--upstream", directory]
    for arguments in [
        training,
        [*training, "--resume", directory],
        ["voice-select-checkpoint", "test-voice", "--checkpoint", directory],
    ]:
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0, result.output
    assert [kind for kind, _, _ in calls] == ["fresh", "resume", "select"]
    assert calls[1][1][2] == Path(directory)
    assert calls[1][1][3] == Path(directory)
    assert calls[2][1][2] == Path(directory)
    smaller = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    for arguments in [
        [*training, "--model", smaller],
        [*training, "--resume", directory, "--model", smaller],
    ]:
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0, result.output
        assert calls[-1][2]["model"] == smaller


def test_cli_prompt_and_input_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("YAPIT_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(
        "yapit.cli.write_script", lambda *args, **kwargs: Script.from_text("Hello.")
    )
    result = CliRunner().invoke(app, ["generate-script", "One weekend", "--seconds", "30"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["beats"][0]["text"] == "Hello."
    with pytest.raises(ValidationError):
        Brief(seconds=float("nan"))
    with pytest.raises(ValidationError):
        Script(title="Bad", beats=[Beat(text="  ")])


def test_extract_reports_ffmpeg_timeout_and_cleans_temporary_files(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    source.touch()
    temporary = []
    monkeypatch.setenv("YAPIT_FFMPEG", "ffmpeg-timeout-test")

    def timeout(command, **kwargs):
        temporary.append(Path(command[-1]))
        temporary[-1].write_bytes(b"partial decoded audio")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("yapit.common_utils.mediautils.subprocess.run", timeout)
    result = CliRunner().invoke(app, ["extract_audio", str(source)])
    assert result.exit_code == 1
    assert "FFmpeg timed out after 600 seconds" in result.output
    assert not temporary[0].parent.exists()
    assert not list(DataStore().audio.iterdir())


@pytest.mark.parametrize(
    "flags,offline", [([], True), (["--offline"], True), (["--no-offline"], False)]
)
def test_cli_passes_settings_and_emits_json(monkeypatch, tmp_path, flags, offline):
    source = tmp_path / "input.mp3"
    source.touch()
    calls = []
    monkeypatch.setattr(
        "yapit.cli.rescript.extract_script",
        lambda media, **kwargs: calls.append((media, kwargs)) or {"status": "needs_review"},
    )
    result = CliRunner().invoke(
        app,
        [
            "extract-script",
            str(source),
            "--model",
            "local-model",
            *flags,
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "needs_review"
    assert calls[0][1]["offline"] is offline
    assert calls[0][1]["model"] == "local-model"


def test_cli_reports_errors(monkeypatch, tmp_path):
    source = tmp_path / "input.mp3"
    source.touch()

    def fail(*args, **kwargs):
        raise ValueError("No speech was detected.")

    monkeypatch.setattr("yapit.cli.rescript.extract_script", fail)
    result = CliRunner().invoke(app, ["extract-script", str(source)])
    assert result.exit_code == 2
    assert "No speech" in result.output


def test_script_and_audio_extraction_commands_are_distinct(monkeypatch, tmp_path):
    source = tmp_path / "source.wav"
    source.touch()
    monkeypatch.setattr(
        "yapit.cli.rescript.extract_script", lambda *args, **kwargs: {"status": "needs_review"}
    )
    runner = CliRunner()
    result = runner.invoke(app, ["extract-script", str(source)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "needs_review"
    monkeypatch.setattr(
        "yapit.rescript.cli.extract_audio_file",
        lambda *args, **kwargs: {"action": "copied", "audio": "data\\audio\\source.wav"},
    )
    result = runner.invoke(app, ["extract_audio", str(source)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["action"] == "copied"


@pytest.mark.parametrize(
    "command,target,arguments",
    [
        ("generate-audio", "yapit.yapper.cli.generate_audio", ["--script"]),
        ("extract-script", "yapit.cli.rescript.extract_script", []),
        ("generate-reel", "yapit.cli.pipeline.generate_reel", ["--audio-path"]),
    ],
)
def test_canonical_commands_dispatch_to_shared_workflows(
    monkeypatch, tmp_path, command, target, arguments
):
    source = tmp_path / "source.json"
    source.write_text(Script.from_text("Hello.").model_dump_json(), encoding="utf-8")
    calls = []

    def operation(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "complete"}

    monkeypatch.setattr(target, operation)
    result = CliRunner().invoke(app, [command, *arguments, str(source)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "complete"
    assert len(calls) == 1
    if command == "generate-reel":
        assert calls[0][1]["audio_path"] == source


def test_canonical_reel_cli_rejects_multiple_sources_before_generation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Ambiguous input must not start generation.")

    monkeypatch.setattr("yapit.cli.pipeline.generate_reel", unexpected)
    result = CliRunner().invoke(
        app, ["generate-reel", "--prompt", "A trip", "--audio-path", "speech.wav"]
    )
    assert result.exit_code == 2, result.output
    assert "exactly one" in result.output


def test_enrollment_cli_reports_app_launch_failure(monkeypatch):
    def fail():
        raise RuntimeError("The enrollment app requires an interactive Windows desktop.")

    monkeypatch.setattr("yapit.enrollment.cli.launcher.enroll_voice", fail)
    result = CliRunner().invoke(app, ["enroll-voice"])
    assert result.exit_code == 1
    assert "interactive Windows desktop" in result.output
