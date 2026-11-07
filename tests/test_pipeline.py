import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
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
    assert abs(len(decoded) / SAMPLE_RATE - 7) < 0.15
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
    assert abs(len(video_audio) / SAMPLE_RATE - result["seconds"]) < 0.15
    assert Path(result["captions"]).read_text().startswith("1\n00:00:00,000 -->")
    assert json.loads(Path(result["provenance"]).read_text())["synthetic_audio"]
    brief = json.loads(Path(result["brief"]).read_text())
    assert brief["words_per_minute"] != 150


def test_no_script_returns_prompt_without_loading_voice(enrolled):
    result = Pipeline(enrolled).reel(Brief(topic="A useful camera review"), "not-enrolled")
    assert result["status"] == "awaiting_script"
    assert "game changer" in result["prompt"]
    assert Path(result["brief"]).exists()
