import json
from pathlib import Path

import pytest
from conftest import tone
from pydantic import ValidationError
from typer.main import get_command
from typer.testing import CliRunner

from yapit.cli import app
from yapit.common_utils.mediautils import write_wav
from yapit.common_utils.modelutils import load_model
from yapit.models import Beat, Brief, Script
from yapit.rescript.cli import app as rescript_app


def test_root_command_help_and_readme_catalog_are_complete():
    command = get_command(app)
    visible = {name for name, item in command.commands.items() if not item.hidden}
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    rows = [
        line.split("`")[1].removeprefix("yapit ")
        for line in readme.splitlines()
        if line.startswith("| `yapit ") and line.split("`")[1].count(" ") == 1
    ]
    assert set(rows) == visible
    assert len(rows) == len(visible)
    runner = CliRunner()
    for name in visible:
        result = runner.invoke(app, [name, "--help"])
        assert result.exit_code == 0, result.output
    assert command.commands["yapper"].hidden and command.commands["rescript"].hidden
    index = readme.split("## Documentation index", 1)[1].split("\n## ", 1)[0]
    table = [line for line in index.splitlines() if line.startswith("|")]
    assert table[0] == "| Document | Dev guide | Purpose |"
    assert all(len(line.strip("|").split("|")) == 3 for line in table)


def test_profiles_is_available_without_a_module_prefix(enrolled):
    result = CliRunner().invoke(app, ["profiles"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == ["test-voice"]


def test_enrollment_cli_launches_the_owned_desktop_module(monkeypatch):
    calls = []
    monkeypatch.setattr("yapit.enrollment.desktop.main", lambda: calls.append(True))
    result = CliRunner().invoke(app, ["enroll-app"])
    assert result.exit_code == 0, result.output
    assert calls == [True]


@pytest.mark.parametrize(
    "entry,arguments",
    [
        (app, ["transcribe-media"]),
        (app, ["rescript", "transcribe-media"]),
        (rescript_app, ["transcribe-media"]),
    ],
)
def test_duplicate_transcription_command_is_removed(entry, arguments, monkeypatch, tmp_path):
    source = tmp_path / "source.wav"
    source.touch()

    def unexpected_transcription(*args, **kwargs):
        pytest.fail("The removed alias must not dispatch recognition.")

    monkeypatch.setattr("yapit.rescript.cli.transcribe", unexpected_transcription)
    result = CliRunner().invoke(entry, [*arguments, str(source)])
    assert result.exit_code == 2, result.output
    assert "No such command" in result.output


def test_training_cli_distinguishes_fresh_resume_and_inference_selection(enrolled, monkeypatch):
    calls = []

    def operation(kind):
        def run(*args, **kwargs):
            calls.append((kind, args, kwargs))
            return {"status": "test"}

        return run

    monkeypatch.setattr("yapit.cli.train", operation("fresh"))
    monkeypatch.setattr("yapit.cli.resume_checkpoint", operation("resume"))
    monkeypatch.setattr("yapit.cli.select_checkpoint", operation("select"))
    directory = str(enrolled.root)
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


@pytest.mark.parametrize(
    "arguments,target",
    [
        (["yapper", "generate", "--text", "Supplied words."], "yapit.yapper.cli.generate_audio"),
        (["rescript", "extract_audio"], "yapit.rescript.cli.extract_audio_file"),
        (["rescript", "transcript"], "yapit.rescript.cli.transcribe"),
    ],
)
def test_old_module_groups_remain_compatible(arguments, target, monkeypatch, tmp_path):
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "complete"}

    monkeypatch.setattr(target, run)
    if arguments[0] == "rescript":
        source = tmp_path / "source.wav"
        source.touch()
        arguments = [*arguments, str(source)]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "complete"
    assert len(calls) == 1


def test_model_factory_supports_local_sdk_without_a_cache_keyword(tmp_path):
    class LocalModel:
        def __init__(self, path, *, device):
            self.path = path
            self.device = device

    path = tmp_path / "local-model.gguf"
    model = load_model(LocalModel, path, cache_parameter=None, device="cpu")
    assert model.path == str(path)
    assert model.device == "cpu"


def test_cli_prompt_and_input_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("YAPIT_WORKSPACE", str(tmp_path))
    result = CliRunner().invoke(app, ["prompt", "--topic", "One weekend", "--seconds", "30"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "awaiting_script"
    with pytest.raises(ValidationError):
        Brief(seconds=float("nan"))
    with pytest.raises(ValidationError):
        Script(title="Bad", beats=[Beat(text="  ")])


def test_style_cli_registration_defaults_consent_but_respects_explicit_refusal(
    enrolled, monkeypatch, tmp_path
):
    monkeypatch.setenv("YAPIT_WORKSPACE", str(enrolled.root))
    source = tmp_path / "lively.wav"
    write_wav(source, tone(4, 260))
    runner = CliRunner()
    arguments = [
        "enroll-style",
        "test-voice",
        "--style",
        "lively-vlog",
        "--audio",
        str(source),
        "--text",
        "What a lovely surprise!",
    ]
    refused = runner.invoke(app, [*arguments, "--no-consent"])
    assert refused.exit_code != 0
    assert "own voice" in refused.output
    accepted = runner.invoke(app, arguments)
    assert accepted.exit_code == 0, accepted.output
    assert json.loads(accepted.stdout)["style"] == "lively-vlog"
