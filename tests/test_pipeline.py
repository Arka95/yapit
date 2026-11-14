from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from yapit.models import Beat, Brief, Cue, Script, Timeline
from yapit.pipeline import Pipeline
from yapit.storage import Workspace


def test_no_script_returns_prompt_without_loading_voice(tmp_path):
    result = Pipeline(Workspace(tmp_path / "workspace")).reel(
        Brief(topic="A useful camera review"), "not-enrolled"
    )
    assert result["status"] == "awaiting_script"
    assert "Acceptable estimated duration: 20.1-39.9 seconds" in result["prompt"]
    assert Path(result["brief"]).exists()


def test_auto_script_uses_text_writer_without_extracting_visual_frames(tmp_path, monkeypatch):
    root = Workspace(tmp_path / "workspace")
    supplied = Script(title="Trip", beats=[Beat(text="A quiet morning.")])
    timeline = Timeline(
        seconds=28,
        cues=[Cue(start=0, end=28, text="A quiet morning.", visual="Road", beat=0)],
    )
    writer = Mock(return_value=supplied)
    frames = Mock(side_effect=AssertionError("Script generation must not extract image frames."))
    monkeypatch.setattr("yapit.pipeline._write_script", writer)
    monkeypatch.setattr("yapit.pipeline.media_frames", frames)
    monkeypatch.setattr(
        "yapit.pipeline.select_voice",
        lambda *args: (SimpleNamespace(words_per_minute=120), tmp_path / "voice"),
    )
    monkeypatch.setattr(
        "yapit.pipeline.render_audio",
        lambda *args, **kwargs: ({"audio": str(tmp_path / "audio.mp3"), "seconds": 28}, timeline),
    )
    monkeypatch.setattr("yapit.pipeline.render_guide", lambda *args, **kwargs: 28.75)
    result = Pipeline(root).reel(
        Brief(topic="A useful trip", facts="One afternoon", seconds=30),
        auto_script=True,
        media=tmp_path / "unused.mp4",
    )
    request = writer.call_args.args[0]
    assert request.words_per_minute == 120 and request.facts == "One afternoon"
    assert writer.call_args.kwargs == {"duration_tolerance": 9.9, "end_buffer": 0.75}
    frames.assert_not_called()
    assert result["video_seconds"] == 28.75
