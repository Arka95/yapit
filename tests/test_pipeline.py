import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest
from conftest import tone

from yapit.common_utils.mediautils import (
    SAMPLE_RATE,
    decode_audio,
)
from yapit.models import Beat, Brief, Script, Timeline
from yapit.pipeline import Pipeline


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


def test_real_mp3_vertical_video_and_contiguous_timeline(enrolled):
    speech = ToneSpeech()
    result = Pipeline(enrolled).reel(Brief(seconds=7), "test-voice", script(), backend=speech)
    assert result["status"] == "complete"
    assert Path(result["audio"]).is_relative_to(enrolled.data.generated_audio)
    assert Path(result["wav"]).is_relative_to(enrolled.data.generated_audio)
    assert Path(result["script"]).is_relative_to(enrolled.data.transcripts)
    assert Path(result["captions"]).is_relative_to(enrolled.data.transcripts)
    assert Path(result["video"]).parent == enrolled.data.generated_video
    assert Path(result["video"]).name.endswith(".guide.mp4")
    assert not list(Path(result["job"]).rglob("*.wav"))
    assert not list(Path(result["job"]).rglob("*.mp3"))
    assert not list(Path(result["job"]).rglob("*.mp4"))
    assert speech.calls == [beat.text for beat in script().beats]
    decoded = decode_audio(Path(result["audio"]))
    assert len(decoded) / SAMPLE_RATE == pytest.approx(6.7, abs=0.03)
    assert result["tempo_factor"] == 1
    assert np.max(np.abs(decoded)) > 0.01
    timeline = Timeline.model_validate_json(Path(result["timeline"]).read_text())
    assert timeline.cues[0].start == 0
    assert timeline.cues[-1].end == result["seconds"]
    reader = imageio_ffmpeg.read_frames(result["video"])
    try:
        metadata = next(reader)
        assert metadata["size"] == (1080, 1920)
        assert metadata["fps"] == 30
        assert len(next(reader)) == 1080 * 1920 * 3
    finally:
        reader.close()
    video_audio = decode_audio(Path(result["video"]))
    assert len(video_audio) / SAMPLE_RATE == pytest.approx(result["video_seconds"], abs=0.05)
    assert result["video_seconds"] - result["seconds"] >= 0.75
    assert np.max(np.abs(video_audio[round((result["seconds"] + 0.1) * SAMPLE_RATE) :])) < 0.001
    assert result["music"] is None
    assert Path(result["captions"]).read_text().startswith("1\n00:00:00,000 -->")
    assert json.loads(Path(result["provenance"]).read_text())["synthetic_audio"]
    brief = json.loads(Path(result["brief"]).read_text())
    assert brief["words_per_minute"] != 150


def test_no_script_returns_prompt_without_loading_voice(enrolled):
    result = Pipeline(enrolled).reel(Brief(topic="A useful camera review"), "not-enrolled")
    assert result["status"] == "awaiting_script"
    assert "game changer" in result["prompt"]
    assert "Acceptable estimated duration: 20.1-39.9 seconds" in result["prompt"]
    assert Path(result["brief"]).exists()


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
