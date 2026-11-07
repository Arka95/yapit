import numpy as np
import pytest
from PIL import Image

from yapit.common_utils.mediautils import (
    SAMPLE_RATE,
    fit_audio,
    media_frames,
    resample,
    run_media,
    write_wav,
)


def tone(seconds: float, frequency: float = 180) -> np.ndarray:
    time = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * time) ** 2
    return (0.15 * envelope * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_shared_resampling_preserves_duration_and_amplitude():
    source = (0.1 * np.sin(2 * np.pi * 220 * np.arange(48000) / 48000)).astype(np.float32)
    result = resample(source, 48000)
    assert len(result) == SAMPLE_RATE
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) == pytest.approx(0.1, abs=0.005)


def test_already_in_tolerance_does_not_force_extreme_stretch(tmp_path):
    source = tmp_path / "raw.wav"
    write_wav(source, tone(6))
    audio, factor = fit_audio(source, tmp_path / "unused.wav", 9)
    assert factor == 1
    assert len(audio) == 6 * SAMPLE_RATE
    assert not (tmp_path / "unused.wav").exists()


@pytest.mark.parametrize(("target", "duration"), [(30, 28), (30, 20.1), (30, 39.9), (180, 239.4)])
def test_video_buffer_preserves_natural_length_at_both_boundaries(tmp_path, target, duration):
    source = tmp_path / "raw.wav"
    write_wav(source, tone(duration))
    audio, factor = fit_audio(
        source, tmp_path / "unused.wav", target, natural_duration_tolerance=target * 0.33
    )
    assert factor == 1
    assert len(audio) == round(duration * SAMPLE_RATE)
    assert not (tmp_path / "unused.wav").exists()


@pytest.mark.parametrize("duration", [20, 40])
def test_video_buffer_rejects_outside_range_without_stretching(tmp_path, duration):
    source = tmp_path / "raw.wav"
    write_wav(source, tone(duration))
    with pytest.raises(ValueError, match="Rewrite"):
        fit_audio(source, tmp_path / "unused.wav", 30, natural_duration_tolerance=9.9)
    assert not (tmp_path / "unused.wav").exists()


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
