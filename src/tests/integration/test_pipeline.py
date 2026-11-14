"""Opt-in pipeline integration checks without full-video rendering."""

from pathlib import Path

import pytest

from yapit.models import Beat, Brief, Script
from yapit.pipeline import Pipeline
from yapit.tests.helpers import tone


class ToneSpeech:
    """Deterministic PCM for pipeline tests. Never presented as a voice clone."""

    def __init__(self, seconds=3.1):
        self.seconds = seconds
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        return tone(self.seconds, 180 + len(self.calls) * 40)


def script():
    return Script(
        title="Weekend",
        beats=[
            Beat(text="Choose a nearby base for the weekend.", pause_ms=200),
            Beat(text="Keep one afternoon free, then enjoy the slower pace.", pause_ms=300),
        ],
    )


def test_video_buffer_reaches_encoding_and_timeline_without_extra_silence(enrolled, monkeypatch):
    rendered = []
    from yapit.editor.mixing import video_seconds

    def render(timeline, *args, **kwargs):
        rendered.append(timeline)
        return video_seconds(timeline, kwargs["mix"])

    monkeypatch.setattr("yapit.pipeline.render_guide", render)
    supplied = Script(title="Natural pace", beats=[Beat(text="A useful thought.", pause_ms=0)])
    result = Pipeline(enrolled).reel(
        Brief(seconds=30), "test-voice", supplied, backend=ToneSpeech(22)
    )
    assert result["seconds"] == pytest.approx(22, abs=0.03)
    assert result["tempo_factor"] == 1
    assert rendered[0].seconds == result["seconds"]
    assert rendered[0].cues[-1].end == result["seconds"]
    assert Script.model_validate_json(Path(result["script"]).read_text()) == supplied


def test_narration_only_pipeline_retains_existing_duration_gate(enrolled):
    supplied = Script(title="Natural pace", beats=[Beat(text="A useful thought.", pause_ms=0)])
    with pytest.raises(ValueError, match="Rewrite"):
        Pipeline(enrolled).reel(
            Brief(seconds=30), "test-voice", supplied, backend=ToneSpeech(22), render_video=False
        )


def test_default_video_never_plans_searches_or_downloads_music(enrolled, monkeypatch):
    from yapit.editor.mixing import video_seconds

    def forbidden(*args, **kwargs):
        raise AssertionError("Default video must not curate or acquire background music.")

    monkeypatch.setattr("yapit.pipeline.plan_music", forbidden)
    monkeypatch.setattr("yapit.musicgen.catalog.search_music", forbidden)
    monkeypatch.setattr("yapit.musicgen.catalog.download_music", forbidden)
    monkeypatch.setattr(
        "yapit.pipeline.render_guide",
        lambda timing, *args, **kwargs: video_seconds(timing, kwargs["mix"]),
    )
    result = Pipeline(enrolled).reel(Brief(seconds=7), "test-voice", script(), backend=ToneSpeech())
    assert result["music"] is None
    assert not list(enrolled.data.background_music.iterdir())
