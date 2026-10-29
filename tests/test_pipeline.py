import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest
from conftest import tone
from PIL import Image

from yapit.media import SAMPLE_RATE, decode_audio, fit_audio, media_frames, run_media, write_wav
from yapit.models import Beat, Brief, Script, Timeline
from yapit.pipeline import Pipeline
from yapit.timeline import build_timeline


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
    assert json.loads((Path(result["job"]) / "provenance.json").read_text())["synthetic_audio"]
    brief = json.loads((Path(result["job"]) / "brief.json").read_text())
    assert brief["words_per_minute"] != 150


def test_narration_only_skips_video(enrolled):
    result = Pipeline(enrolled).reel(
        Brief(seconds=7),
        "test-voice",
        script(),
        backend=ToneSpeech(),
        render_video=False,
    )
    assert Path(result["audio"]).exists()
    assert result["video"] is None
    assert not (Path(result["job"]) / "video").exists()


def test_duration_failure_is_persisted_without_truncating_speech(enrolled):
    with pytest.raises(ValueError, match="Rewrite"):
        Pipeline(enrolled).reel(
            Brief(seconds=5),
            "test-voice",
            script(),
            backend=ToneSpeech(8),
        )
    status_files = list((enrolled.root / "outputs").glob("*/status.json"))
    assert len(status_files) == 1
    status = json.loads(status_files[0].read_text())
    assert status["status"] == "failed"
    assert not (status_files[0].parent / "narration.mp3").exists()


def test_no_script_returns_prompt_without_loading_voice(enrolled):
    result = Pipeline(enrolled).reel(Brief(topic="A useful camera review"), "not-enrolled")
    assert result["status"] == "awaiting_script"
    assert "game changer" in result["prompt"]
    assert Path(result["job"], "brief.json").exists()


def test_already_in_tolerance_does_not_force_extreme_stretch(tmp_path):
    source = tmp_path / "raw.wav"
    write_wav(source, tone(6))
    audio, factor = fit_audio(source, tmp_path / "unused.wav", 9)
    assert factor == 1
    assert len(audio) == 6 * SAMPLE_RATE
    assert not (tmp_path / "unused.wav").exists()


def test_long_beat_prefers_quiet_window_and_keeps_cues_contiguous():
    audio = tone(8)
    audio[round(3.75 * SAMPLE_RATE) : round(3.85 * SAMPLE_RATE)] = 0
    one = Script(title="Long beat", beats=[Beat(text="One two three four five six seven eight.")])
    timeline = build_timeline(one, [8], 8, audio)
    assert 3.75 <= timeline.cues[0].end <= 3.85
    assert timeline.cues[0].end == timeline.cues[1].start
    assert timeline.cues[-1].end == 8
    with pytest.raises(ValueError, match="positive"):
        build_timeline(one, [0], 8)


def test_image_and_video_context_are_local_bounded_snapshots(tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (2000, 1000), "blue").save(image)
    frames = media_frames(image, tmp_path / "image-frames")
    with Image.open(frames[0]) as converted:
        assert converted.size == (1280, 640)
    video = tmp_path / "source.mp4"
    run_media(
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x90:r=2:d=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ]
    )
    snapshots = media_frames(video, tmp_path / "video-frames")
    assert 1 <= len(snapshots) <= 6
    assert all(path.suffix == ".jpg" for path in snapshots)
