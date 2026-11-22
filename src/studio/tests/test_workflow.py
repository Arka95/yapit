import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from yapit.models import Cue, Timeline
from yapit.storage import Workspace
from yapit.studio import workflow


@pytest.fixture
def studio(tmp_path, monkeypatch):
    monkeypatch.setenv("YAPIT_DATA_DIR", str(tmp_path / "data"))
    session = workflow.StudioSession(Workspace(tmp_path / "workspace"))
    monkeypatch.setattr(workflow, "decode_audio", Mock(return_value=np.zeros(24000)))
    monkeypatch.setattr(workflow, "write_wav", lambda path, wave: path.touch())
    yield session
    session.close()


def audio_file(tmp_path):
    path = tmp_path / "narration.mp3"
    path.touch()
    return path


def timeline_file(tmp_path, seconds=1):
    timeline = Timeline(
        seconds=seconds,
        cues=[Cue(start=0, end=seconds, text="Hello.", visual="A view", beat=0)],
    )
    path = tmp_path / "timeline.json"
    path.write_text(timeline.model_dump_json())
    return path


def test_loading_external_audio_prepares_playback_without_recognition(
    studio, tmp_path, monkeypatch
):
    recognize = Mock(side_effect=AssertionError("Loading must not run a model."))
    monkeypatch.setattr("yapit.rescript.transcription.transcribe", recognize)
    preview = studio.load_audio(audio_file(tmp_path))
    assert preview.timeline is None
    assert preview.seconds == 1
    assert preview.playback.is_file()
    workflow.decode_audio.assert_called_once_with(preview.audio, max_seconds=180)
    recognize.assert_not_called()


def test_generated_audio_reuses_its_timeline_and_renders_without_tts(studio, tmp_path, monkeypatch):
    audio = audio_file(tmp_path)
    preview = studio.load_audio(audio, timeline_file(tmp_path))
    render = Mock(return_value={"video": str(tmp_path / "guide.mp4")})
    recognize = Mock(side_effect=AssertionError("Generated timing needs no ASR."))
    monkeypatch.setattr("yapit.editor.generate_video", render)
    monkeypatch.setattr("yapit.rescript.transcription.transcribe", recognize)
    assert studio.analyze_beats() is preview
    result = studio.generate_video()
    render.assert_called_once_with(
        preview.timeline, audio, workspace=studio.workspace, actual_seconds=1
    )
    assert studio.video == Path(result["video"])
    recognize.assert_not_called()


def test_bad_timeline_preserves_loaded_audio_and_replacement_clears_video(studio, tmp_path):
    old = studio.load_audio(audio_file(tmp_path))
    studio.video = tmp_path / "old.mp4"
    with pytest.raises(ValueError, match="duration"):
        studio.load_audio(old.audio, timeline_file(tmp_path, seconds=2))
    assert studio.preview is old and old.playback.exists()
    assert studio.video is not None
    replacement = studio.load_audio(old.audio)
    assert studio.preview is replacement
    assert not old.playback.exists()
    assert studio.video is None


def test_imported_beats_use_offline_recognition_once(studio, tmp_path, monkeypatch):
    preview = studio.load_audio(audio_file(tmp_path))
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps({"segments": [{"start": 0.1, "end": 0.9, "text": "Hi."}]}))
    recognize = Mock(return_value={"transcript": str(transcript), "source_seconds": 1})
    monkeypatch.setattr("yapit.rescript.transcription.transcribe", recognize)
    analyzed = studio.analyze_beats()
    assert analyzed.timeline.cues[0].text == "Hi."
    assert analyzed.timeline.cues[0].start == 0
    assert analyzed.playback == preview.playback
    assert studio.analyze_beats() is analyzed
    recognize.assert_called_once_with(
        preview.audio,
        data=studio.workspace.data,
        max_seconds=180,
        offline=True,
        export_scripts=False,
    )


def test_render_requires_reviewable_beats_and_cleanup_only_removes_owned_preview(studio, tmp_path):
    with pytest.raises(ValueError, match="Analyze narration beats"):
        studio.generate_video()
    preview = studio.load_audio(audio_file(tmp_path))
    with pytest.raises(ValueError, match="Analyze narration beats"):
        studio.generate_video()
    studio.close()
    assert not preview.playback.exists()
    assert preview.audio.exists()


def test_waveform_keeps_short_transients_and_short_audio():
    assert workflow.waveform_peaks(np.array([0, -0.8, 0.3, 0]), bins=2) == [0.8, 0.3]
    assert workflow.waveform_peaks(np.array([-0.5])) == [0.5]


def test_backend_discovery_lists_plugins_without_loading_them(monkeypatch):
    plugin = SimpleNamespace(name="trusted-plugin", load=Mock())
    monkeypatch.setattr(workflow.importlib.metadata, "entry_points", lambda **kwargs: [plugin])
    assert {"qwen", "pocket", "trusted-plugin"} <= set(workflow.backend_names())
    plugin.load.assert_not_called()
