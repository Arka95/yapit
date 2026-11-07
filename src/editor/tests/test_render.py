import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image

from yapit.common_utils.mediautils import SAMPLE_RATE, decode_audio, write_wav
from yapit.editor import render_guide
from yapit.models import Cue, Timeline


def test_coloured_clips_follow_frame_durations_and_publish_complete_video(tmp_path):
    durations = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    colours = [
        (255, 0, 0),
        (0, 255, 255),
        (0, 255, 0),
        (0, 100, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]
    cues = []
    start = 0
    for index, duration in enumerate(durations):
        cues.append(
            Cue(start=start, end=start + duration, text=f"Clip {index}", visual="Footage", beat=0)
        )
        start += duration
    timeline = Timeline(seconds=start, width=160, height=256, fps=30, cues=cues)
    time = np.arange(round(start * SAMPLE_RATE)) / SAMPLE_RATE
    audio = tmp_path / "narration.wav"
    write_wav(audio, (0.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32))
    work = tmp_path / "workspace"
    video = tmp_path / "data" / "generated_video" / "trip.guide.mp4"
    captions = tmp_path / "data" / "transcripts" / "trip.captions.srt"
    render_guide(timeline, audio, work, video=video, captions=captions, prefix="trip")
    assert video.is_file()
    assert not (work / "trip.guide.mp4").exists()
    assert "Clip 6" in captions.read_text()
    for index, colour in enumerate(colours, start=1):
        with Image.open(work / f"trip.clip-{index:03}.png") as slate:
            assert slate.getpixel((0, 0)) == colour
    expected = np.repeat(np.asarray(colours), [round(d * timeline.fps) for d in durations], axis=0)
    reader = imageio_ffmpeg.read_frames(str(video))
    try:
        metadata = next(reader)
        assert metadata["fps"] == timeline.fps
        pixels = [
            np.frombuffer(frame, dtype=np.uint8).reshape(timeline.height, timeline.width, 3)[0, 0]
            for frame in reader
        ]
    finally:
        reader.close()
    assert len(pixels) == len(expected)
    assert np.max(np.abs(np.asarray(pixels, dtype=int) - expected)) <= 5
    assert len(decode_audio(video)) / SAMPLE_RATE == pytest.approx(start, abs=0.05)


def test_render_failure_does_not_publish_video(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("encoding failed")

    monkeypatch.setattr("yapit.editor.render.run_media", fail)
    timeline = Timeline(
        seconds=5, cues=[Cue(start=0, end=5, text="Test", visual="Footage", beat=0)]
    )
    video = tmp_path / "data" / "generated_video" / "failed.mp4"
    with pytest.raises(RuntimeError, match="encoding failed"):
        render_guide(
            timeline,
            tmp_path / "audio.wav",
            tmp_path / "workspace",
            video=video,
            captions=tmp_path / "captions.srt",
        )
    assert not video.exists()


def test_prefix_cannot_inject_paths_or_playlist_directives(tmp_path):
    timeline = Timeline(
        seconds=5, cues=[Cue(start=0, end=5, text="Test", visual="Footage", beat=0)]
    )
    with pytest.raises(ValueError, match="simple ASCII filename"):
        render_guide(
            timeline,
            tmp_path / "audio.wav",
            tmp_path / "workspace",
            video=tmp_path / "video.mp4",
            prefix="trip'\nfile 'outside",
        )
    assert not (tmp_path / "workspace").exists()
