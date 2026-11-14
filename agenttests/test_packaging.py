import json
import subprocess
import sys
from importlib.resources import files
from pathlib import Path


def test_root_command_help_and_readme_catalog_are_complete():
    from typer.main import get_command
    from typer.testing import CliRunner

    from yapit.cli import app

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


def test_package_import_and_packaged_config_do_not_initialize_server_or_model_sdks():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import yapit.mcp; "
            "assert not set(sys.modules).intersection("
            "{'mcp', 'yapit.mcp.server', 'torch', 'qwen_tts', 'faster_whisper'})",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    config = json.loads(files("yapit.mcp").joinpath("examples", "mcp.config.json").read_text())
    assert config["mcpServers"]["yapit"]["args"][2].endswith(r"\yapit.cmd")


def test_module_import_does_not_initialize_training_sdks_or_models(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import yapit.yapper; "
            "assert 'yapit.training' not in sys.modules; "
            "import yapit.training; "
            "assert callable(yapit.training.train); "
            "assert not set(sys.modules).intersection({'torch', 'flash_attn', 'huggingface_hub'})",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "models").exists()


def test_prompt_template_retains_editorial_guidance():
    from yapit.models import Brief
    from yapit.yapper.script_planner import build_prompt

    prompt = build_prompt(Brief(topic="Test", seconds=30))
    assert "game changer" in prompt
    assert "Do not add filler, inflate pauses" in prompt
