from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from yapit.common_utils.fileutils import save_json
from yapit.enrollment import EnrollmentSession, enroll, load_voice, profile_names
from yapit.enrollment.desktop import RecordingPanel
from yapit.models import Sample, Voice
from yapit.storage import DataStore


@pytest.fixture
def reference_session(tmp_path, monkeypatch):
    def write(path, data):
        Path(path).write_bytes(np.asarray(data, dtype=np.float32).tobytes())

    def decode(path, *, max_seconds):
        data = np.frombuffer(Path(path).read_bytes(), dtype=np.float32).copy()
        if not len(data) or len(data) > max_seconds * 1000:
            raise ValueError("Audio is empty or longer than the recording limit.")
        return data

    for module in ("yapit.enrollment.session", "yapit.enrollment.profiles"):
        monkeypatch.setattr(f"{module}.SAMPLE_RATE", 1000)
        monkeypatch.setattr(f"{module}.write_wav", write)
        monkeypatch.setattr(f"{module}.decode_audio", decode)
    session = EnrollmentSession(DataStore(tmp_path / "data"), "myvoice")
    return session


def recording(session, seconds=60, level=0.1):
    path = session.new_take()
    path.write_bytes(np.full(seconds * 1000, level, dtype=np.float32).tobytes())
    return path


def test_one_confirmed_paragraph_resumes_and_publishes_only_one_reference(reference_session):
    session = reference_session
    assert len(session.draft.readings) == 1
    assert not session.ready
    assert profile_names(session.store) == ["myvoice"]
    raw = recording(session)
    with pytest.raises(ValueError, match="confirm"):
        session.save_take(raw, False, True)
    session.save_take(raw, True, True)
    assert not session.store.voice_path(session.alias).exists()
    resumed = EnrollmentSession(session.store, session.alias)
    assert resumed.ready
    voice = resumed.finish()
    assert len(voice.samples) == 1
    assert voice.reference_audio == voice.samples[0].audio
    assert voice.reference_text == session.draft.readings[0].text
    assert (session.audio_folder / voice.reference_audio).is_file()
    assert not raw.exists()
    assert EnrollmentSession(session.store, session.alias).draft.completed


@pytest.mark.parametrize("seconds,level,reason", [(10, 0.1, "45"), (60, 1, "clipped")])
def test_invalid_reference_cannot_complete_enrollment(reference_session, seconds, level, reason):
    session = reference_session
    with pytest.raises(ValueError, match=reason):
        session.save_take(recording(session, seconds, level), True, False)
    assert not session.ready
    assert not session.store.voice_path(session.alias).exists()


def test_multi_clip_import_rejected_without_changing_existing_profile(reference_session, tmp_path):
    session = reference_session
    session.save_take(recording(session), True, True)
    session.finish()
    original = session.store.voice_path(session.alias).read_bytes()
    manifest = tmp_path / "multiple.json"
    save_json(
        manifest,
        {"samples": [{"audio": "a.wav", "text": "First"}, {"audio": "b.wav", "text": "Second"}]},
    )
    with pytest.raises(ValueError, match="one complete"):
        enroll(session.store, session.alias, manifest)
    assert session.store.voice_path(session.alias).read_bytes() == original


def test_reenrollment_preserves_legacy_dataset_and_clears_stale_conditioning(reference_session):
    from yapit.training.dataset import load_training_samples

    session = reference_session
    session.audio_folder.mkdir()
    samples = []
    originals = {}
    for index in range(16):
        path = session.audio_folder / f"old-{index}.wav"
        path.write_bytes(f"legacy recording {index}".encode())
        originals[path] = path.read_bytes()
        samples.append(Sample(audio=path.name, text=f"Legacy transcript {index}."))
    voice = Voice(
        name=session.alias,
        consent=True,
        reference_audio=samples[0].audio,
        reference_text=samples[0].text,
        samples=samples,
        words_per_minute=150,
        checkpoint="existing-checkpoint",
        speaker=session.alias,
        pocket_voice_state="old-state.safetensors",
        writing_style="relaxed",
    )
    save_json(session.store.voice_path(session.alias), voice)
    session.save_take(recording(session), True, True)
    result = session.finish()
    assert len(result.samples) == 1
    assert len(load_training_samples(session.store, session.alias)) == 16
    assert result.checkpoint is result.speaker is None
    assert "pocket_voice_state" not in result.model_dump()
    assert result.writing_style == "relaxed"
    assert all(path.read_bytes() == contents for path, contents in originals.items())


def test_saved_replacement_keeps_published_reference_until_finish(reference_session):
    session = reference_session
    session.save_take(recording(session), True, False)
    first = session.finish()
    before = (session.audio_folder / first.reference_audio).read_bytes()
    session.select_sample(0)
    session.save_take(recording(session, level=0.2), True, False)
    assert load_voice(session.store, session.alias).reference_audio == first.reference_audio
    second = session.finish()
    assert second.reference_audio != first.reference_audio
    assert (session.audio_folder / first.reference_audio).read_bytes() == before


def test_failed_microphone_start_closes_log_and_enrollment_cleans_it(
    reference_session, monkeypatch
):
    from yapit.enrollment.recorder import MicrophoneRecorder

    recorder = MicrophoneRecorder()
    monkeypatch.setattr("yapit.enrollment.recorder.os", SimpleNamespace(name="nt"))
    monkeypatch.setattr("yapit.enrollment.recorder.ffmpeg", lambda: "mock-ffmpeg")
    monkeypatch.setattr("yapit.enrollment.recorder.subprocess.CREATE_NO_WINDOW", 0, raising=False)
    monkeypatch.setattr(
        "yapit.enrollment.recorder.subprocess.Popen",
        Mock(side_effect=FileNotFoundError("Mock startup failure")),
    )
    with pytest.raises(FileNotFoundError):
        recorder.prepare("Mock microphone", reference_session.scratch)
    assert recorder.log.closed
    assert not recorder.ready
    log = recorder.log_path
    reference_session.save_take(recording(reference_session), True, False)
    reference_session.finish()
    assert not log.exists()


@pytest.mark.parametrize("training", [False, True])
def test_recording_panel_save_publishes_enrollment_but_only_collects_training(training):
    panel = RecordingPanel.__new__(RecordingPanel)
    panel.app = SimpleNamespace(busy=False)
    panel.training, panel.recording, panel.take_ready = training, False, True
    panel.session, panel.recorder = Mock(), Mock()
    panel.take = Path("take.wav")
    panel.confirmed = SimpleNamespace(get=lambda: True)
    panel.normalize = SimpleNamespace(get=lambda: True)
    panel.status = Mock()
    panel.stop_playback = Mock()
    panel.work = Mock()
    panel.save_next()
    operation, callback = panel.work.call_args.args
    operation()
    panel.session.save_take.assert_called_once_with(Path("take.wav"), True, True)
    assert panel.session.finish.call_count == (0 if training else 1)
    assert callback == (panel.take_saved if training else panel.enrollment_finished)


@pytest.mark.parametrize("limit", [20, 90])
def test_recorder_caps_at_its_workflow_limit_and_flushes_without_starting_hardware(limit):
    from yapit.common_utils.mediautils import SAMPLE_RATE
    from yapit.enrollment.recorder import MicrophoneRecorder

    recorder = MicrophoneRecorder(max_seconds=limit)
    writer, stream = Mock(), Mock()
    recorder._writer, recorder._file = writer, stream
    recorder._frames = limit * SAMPLE_RATE - 5
    recorder._write(b"\0" * 20)
    writer.writeframes.assert_called_once_with(b"\0" * 10)
    writer.close.assert_called_once_with()
    stream.close.assert_called_once_with()
    assert recorder.captured_seconds == limit
