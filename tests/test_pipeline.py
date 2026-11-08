from pathlib import Path

from yapit.models import Brief
from yapit.pipeline import Pipeline
from yapit.storage import Workspace


def test_no_script_returns_prompt_without_loading_voice(tmp_path):
    result = Pipeline(Workspace(tmp_path / "workspace")).reel(
        Brief(topic="A useful camera review"), "not-enrolled"
    )
    assert result["status"] == "awaiting_script"
    assert "Acceptable estimated duration: 20.1-39.9 seconds" in result["prompt"]
    assert Path(result["brief"]).exists()
