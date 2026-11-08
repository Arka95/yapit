from unittest.mock import Mock

import pytest

from yapit.editor import RenderSettings, render_guide
from yapit.editor.render import _encode_video, _prepare_slates
from yapit.editor.templates import DEFAULT_STYLE
from yapit.models import Cue, Timeline


@pytest.fixture
def timeline():
    return Timeline(seconds=5, cues=[Cue(start=0, end=5, text="Test", visual="Footage", beat=0)])


def test_render_failure_does_not_publish_video(tmp_path, monkeypatch, timeline):
    monkeypatch.setattr(
        "yapit.editor.render.prepare_audio_inputs", lambda *args, **kwargs: ([], 5.75)
    )
    monkeypatch.setattr(
        "yapit.editor.render._prepare_slates",
        lambda *args, **kwargs: (tmp_path / "slates.txt", "captions"),
    )
    monkeypatch.setattr(
        "yapit.editor.render.run_media", Mock(side_effect=RuntimeError("encoding failed"))
    )
    video = tmp_path / "failed.mp4"
    with pytest.raises(RuntimeError, match="encoding failed"):
        render_guide(
            timeline,
            tmp_path / "audio.wav",
            tmp_path / "workspace",
            video=video,
            captions=tmp_path / "captions.srt",
        )
    assert not video.exists()


def test_publish_encoded_artifact_without_rendering(tmp_path, monkeypatch, timeline):
    monkeypatch.setattr(
        "yapit.editor.render.prepare_audio_inputs", lambda *args, **kwargs: ([], 5.75)
    )
    monkeypatch.setattr(
        "yapit.editor.render._prepare_slates",
        lambda *args, **kwargs: (tmp_path / "slates.txt", "captions"),
    )

    def fake_encode(timeline, audio, playlist, rendered, **kwargs):
        rendered.write_bytes(b"encoded artifact")

    monkeypatch.setattr("yapit.editor.render._encode_video", fake_encode)
    video = tmp_path / "published" / "video.mp4"
    captions = tmp_path / "captions.srt"
    seconds = render_guide(
        timeline,
        tmp_path / "audio.wav",
        tmp_path / "workspace",
        video=video,
        captions=captions,
    )
    assert video.read_bytes() == b"encoded artifact"
    assert captions.read_text() == "captions"
    assert seconds == 5.75


def test_prefix_cannot_inject_paths_or_playlist_directives(tmp_path, timeline):
    with pytest.raises(ValueError, match="simple ASCII filename"):
        render_guide(
            timeline,
            tmp_path / "audio.wav",
            tmp_path / "workspace",
            video=tmp_path / "video.mp4",
            prefix="trip'\nfile 'outside",
        )
    assert not (tmp_path / "workspace").exists()


def test_encoding_uses_supplied_quality_settings(tmp_path, monkeypatch, timeline):
    media = Mock()
    monkeypatch.setattr("yapit.editor.render.run_media", media)
    _encode_video(
        timeline,
        tmp_path / "audio.wav",
        tmp_path / "slates.txt",
        tmp_path / "video.mp4",
        inputs=[],
        graph="[1:a]anull[mixed]",
        seconds=5.75,
        settings=RenderSettings(preset="veryfast", crf=28, audio_bitrate="128k"),
    )
    args = media.call_args.args[0]
    assert args[args.index("-preset") + 1] == "veryfast"
    assert args[args.index("-crf") + 1] == "28"
    assert args[args.index("-b:a") + 1] == "128k"


def test_slate_plan_extends_last_visual_but_not_captions(tmp_path, monkeypatch, timeline):
    placeholder = Mock(return_value=Mock())
    monkeypatch.setattr("yapit.editor.render.create_placeholder", placeholder)
    playlist, captions = _prepare_slates(
        timeline, tmp_path, stem="", seconds=5.766666667, style=DEFAULT_STYLE
    )
    assert placeholder.call_args.args[1].end == pytest.approx(5.766666667)
    assert "00:00:05,000" in captions
    durations = [
        float(line.split()[1])
        for line in playlist.read_text().splitlines()
        if line.startswith("duration ")
    ]
    assert durations == pytest.approx([5.766666667])
