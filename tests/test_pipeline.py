from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from yapit.common_utils.fileutils import save_json
from yapit.models import Beat, Brief, Cue, Script, Timeline
from yapit.pipeline import Pipeline, generate_reel
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


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"prompt": ""},
        {"prompt": "   "},
        {"prompt": "Trip", "audio_path": Path("speech.wav")},
        {"script": Script.from_text("Hello."), "prompt": "Trip"},
        {"script": Script.from_text("Hello."), "audio_path": Path("speech.wav")},
    ],
)
def test_reel_rejects_missing_blank_or_ambiguous_source(arguments):
    with pytest.raises(ValueError, match="exactly one|blank"):
        generate_reel(**arguments)


@pytest.mark.parametrize("use_prompt", [False, True])
def test_canonical_reel_routes_text_sources_to_existing_pipeline(tmp_path, monkeypatch, use_prompt):
    root = Workspace(tmp_path / "workspace")
    operation = Mock(return_value={"status": "complete"})
    monkeypatch.setattr(Pipeline, "reel", operation)
    script = Script.from_text("Hello.")
    generate_reel(
        workspace=root,
        **({"prompt": "A quiet trip"} if use_prompt else {"script": script}),
    )
    assert operation.call_args.kwargs["auto_script"] is use_prompt
    assert operation.call_args.kwargs["script"] == (None if use_prompt else script)
    assert operation.call_args.kwargs["voice_name"] == "testuser"
    assert operation.call_args.args[0].seconds == 30


def test_audio_reel_uses_source_timing_without_loading_voice_or_resynthesizing(
    tmp_path, monkeypatch
):
    root = Workspace(tmp_path / "workspace")
    source = tmp_path / "speech.wav"
    source.write_bytes(b"original recording")
    transcript = root.data.transcripts / "speech.transcript.json"
    save_json(
        transcript,
        {
            "segments": [
                {"start": 1, "end": 3, "text": "First thought."},
                {"start": 5, "end": 7, "text": "Last thought."},
            ]
        },
    )
    recognize = Mock(
        return_value={"transcript": str(transcript), "source_seconds": 9, "scripts": []}
    )
    monkeypatch.setattr("yapit.pipeline.transcribe", recognize)
    select_voice = Mock(side_effect=AssertionError("Audio input must not need a voice."))
    synthesize = Mock(side_effect=AssertionError("Audio input must not synthesize speech."))
    monkeypatch.setattr("yapit.pipeline.select_voice", select_voice)
    monkeypatch.setattr("yapit.pipeline.render_audio", synthesize)
    status = root.root / "outputs" / "audio.status.json"
    status.parent.mkdir()
    render = Mock(return_value={"status": "complete", "status_file": str(status)})
    monkeypatch.setattr("yapit.pipeline.generate_video", render)
    result = generate_reel(audio_path=source, workspace=root, voice="not-enrolled", seconds=30)
    timeline, rendered_audio = render.call_args.args
    assert rendered_audio == source
    assert source.read_bytes() == b"original recording"
    assert timeline.seconds == 9
    assert [(cue.start, cue.end) for cue in timeline.cues] == [(0, 5), (5, 9)]
    assert [cue.text for cue in timeline.cues] == ["First thought.", "Last thought."]
    assert render.call_args.kwargs["actual_seconds"] == 9
    assert recognize.call_args.kwargs["max_seconds"] == 180
    assert recognize.call_args.kwargs["data"] is root.data
    select_voice.assert_not_called()
    synthesize.assert_not_called()
    assert result["review_required"] is True
    assert status.exists()
