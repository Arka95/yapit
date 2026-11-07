import json
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from yapit.enrollment import (
    EnrollmentSession,
    MicrophoneRecorder,
    Reading,
)
from yapit.enrollment.profiles import load_voice
from yapit.enrollment.tests.conftest import tone
from yapit.media import SAMPLE_RATE, decode_audio, ffmpeg, write_wav
from yapit.storage import DataStore


@pytest.fixture
def session(tmp_path, monkeypatch):
    readings = [
        Reading(title=f"Passage {index}", text=f"Read passage {index} at a natural pace.")
        for index in range(1, 7)
    ]
    monkeypatch.setattr("yapit.enrollment.session.load_readings", lambda: readings)
    return EnrollmentSession(DataStore(tmp_path / "data"), "my-voice")


def take(session, seconds=4, frequency=200, volume=1):
    path = session.new_take()
    write_wav(path, tone(seconds, frequency) * volume)
    return path


def test_existing_profile_opens_without_consent_and_alias_escape_rejected(enrolled):
    assert EnrollmentSession(enrolled, "TEST-VOICE").ready
    with pytest.raises(ValueError, match="Name"):
        EnrollmentSession(enrolled, "..\\escape")


def test_saved_text_resume_and_final_enrollment(session):
    raw = take(session)
    original = raw.read_bytes()
    with pytest.raises(ValueError, match="confirm"):
        session.save_take(raw, confirmed=False, normalize=True)
    first = session.save_take(raw, confirmed=True, normalize=True)
    assert first["quality"]["reasons"] == []
    assert raw.is_relative_to(session.store.root / session.alias / "audio")
    assert session.folder == session.store.root / session.alias
    assert raw.read_bytes() == original
    resumed = EnrollmentSession(session.store, session.alias)
    assert resumed.index == 1
    assert resumed.draft.samples[0].text == resumed.draft.readings[0].text
    assert len(json.loads((resumed.folder / "my-voice.samples.json").read_text())["samples"]) == 1
    with pytest.raises(ValueError, match="at least 5"):
        resumed.finish()
    for index in range(1, 5):
        resumed.save_take(take(resumed, frequency=200 + index * 30), True, True)
    assert resumed.ready
    assert not resumed.finished
    voice = resumed.finish()
    assert voice.name == "my-voice"
    assert len(voice.samples) == 5
    audio_folder = session.store.voice_audio_dir(voice.name)
    assert audio_folder == session.folder / "audio"
    assert all((audio_folder / sample.audio).exists() for sample in voice.samples)
    assert len(list(audio_folder.glob("*.raw.*.wav"))) == 5
    assert (session.folder / "my-voice.samples.json").is_file()
    assert load_voice(session.store, voice.name).consent
    assert EnrollmentSession(session.store, voice.name).ready


def test_workspace_recording_is_not_an_enrollment_input(session):
    outside = session.store.root.parent / "workspace" / "enrollment" / session.alias
    outside.mkdir(parents=True)
    recording = outside / "sample.wav"
    write_wav(recording, tone(4))
    with pytest.raises(ValueError, match="inside"):
        session.save_take(recording, True, True)
    assert session.index == 0
    assert session.path.is_relative_to(session.store.root)


def test_low_level_cleanup_does_not_change_raw_and_silence_never_passes(session):
    raw = session.new_take()
    data = np.concatenate(
        [
            np.zeros(4 * SAMPLE_RATE),
            tone(4) * 0.3,
            np.zeros(4 * SAMPLE_RATE),
        ]
    )
    write_wav(raw, data)
    with pytest.raises(ValueError, match="quiet"):
        session.save_take(raw, confirmed=True, normalize=False)
    report = session.save_take(raw, confirmed=True, normalize=True)
    assert report["gain_db"] > 20
    assert report["trim_start_seconds"] > 0
    assert report["quality"]["reasons"] == []
    assert len(decode_audio(raw)) == 12 * SAMPLE_RATE
    silent = session.new_take()
    write_wav(silent, np.zeros(4 * SAMPLE_RATE))
    with pytest.raises(ValueError, match="sustained"):
        session.save_take(silent, confirmed=True, normalize=True)
    assert session.index == 1


def test_duplicates_short_clipped_and_outside_takes_rejected(session, tmp_path):
    session.save_take(take(session), confirmed=True, normalize=True)
    with pytest.raises(ValueError, match="duplicates"):
        session.save_take(take(session), confirmed=True, normalize=True)
    with pytest.raises(ValueError, match="at least 3"):
        session.save_take(take(session, seconds=1), confirmed=True, normalize=True)
    clipped = session.new_take()
    write_wav(clipped, np.ones(4 * SAMPLE_RATE))
    with pytest.raises(ValueError, match="clipped"):
        session.save_take(clipped, confirmed=True, normalize=True)
    outside = tmp_path / "outside.wav"
    write_wav(outside, tone(4))
    with pytest.raises(ValueError, match="inside"):
        session.save_take(outside, confirmed=True, normalize=True)
    assert session.index == 1


@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows process control")
def test_record_stop_flushes_real_wav_without_accessing_a_microphone(session, monkeypatch):
    original_popen = subprocess.Popen

    def synthetic_capture(command, **kwargs):
        if "-f" not in command or command[command.index("-f") + 1] != "dshow":
            return original_popen(command, **kwargs)
        assert command[command.index("-f") + 1] == "dshow"
        assert command[command.index("-t") + 1] == "20"
        assert kwargs["stdin"] == subprocess.PIPE
        return original_popen(
            [
                ffmpeg(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-re",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=220:sample_rate=24000",
                "-t",
                "20",
                "-c:a",
                "pcm_s16le",
                command[-1],
            ],
            **kwargs,
        )

    monkeypatch.setattr("yapit.enrollment.recorder.subprocess.Popen", synthetic_capture)
    recorder = MicrophoneRecorder()
    output = session.new_take()
    recorder.start("Test device; never opened", output)
    try:
        time.sleep(1.3)
        assert recorder.stop() == output
        assert recorder.process.poll() == 0
        samples = decode_audio(output)
        assert 0.2 < len(samples) / SAMPLE_RATE < 4
    finally:
        if recorder.process.poll() is None:
            recorder.process.kill()
            recorder.process.wait(timeout=3)


@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows GUI")
def test_real_tk_wizard_advances_after_saving_without_opening_microphone(session, monkeypatch):
    import tkinter as tk

    from yapit.enrollment.desktop import EnrollmentApp

    monkeypatch.setattr("yapit.enrollment.desktop.microphone_devices", lambda: ["Test microphone"])
    root = tk.Tk()
    root.withdraw()
    app = EnrollmentApp(root, session.store)
    errors = []
    playback = []
    notices = []
    monkeypatch.setattr(
        "yapit.enrollment.desktop.messagebox.showerror",
        lambda *args, **kwargs: errors.append(args),
    )
    monkeypatch.setattr(
        "yapit.enrollment.desktop.messagebox.showinfo",
        lambda *args, **kwargs: notices.append(args),
    )
    monkeypatch.setattr("winsound.PlaySound", lambda *args: playback.append(args))

    class FakeRecorder:
        def __init__(self):
            self.process = None
            self.count = 0

        def start(self, device, output):
            assert device == "Test microphone"
            self.output = output
            self.count += 1
            write_wav(output, tone(4, 200 + self.count * 30))
            self.process = SimpleNamespace(poll=lambda: None)

        def stop(self):
            self.process = SimpleNamespace(poll=lambda: 0)
            return self.output

    app.recorder = FakeRecorder()

    def pump_until(predicate):
        deadline = time.monotonic() + 10
        while not predicate():
            if time.monotonic() > deadline:
                raise AssertionError("Desktop action did not finish.")
            root.update()
            time.sleep(0.02)
        root.update()

    try:
        pump_until(lambda: not app.busy)
        app.alias.set(session.alias)
        app.open_button.invoke()
        assert not hasattr(app, "consent_check")
        assert "passage 1 of 6" in app.heading.get()
        assert app.passage.get("1.0", "end").strip() == session.draft.readings[0].text
        assert app.recorder.count == 0
        app.record_button.invoke()
        assert app.recording
        app.stop_button.invoke()
        pump_until(lambda: not app.busy)
        assert not app.recording
        app.play_button.invoke()
        assert playback[-1][0] == str(app.take)
        app.confirmed.set(True)
        app.refresh_controls()
        app.save_button.invoke()
        pump_until(lambda: not app.busy)
        assert not errors
        assert "passage 2 of 6" in app.heading.get()
        assert app.take is None
        assert not app.confirmed.get()
        assert json.loads(app.session.path.read_text())["samples"][0]["text"] == (
            session.draft.readings[0].text
        )
        assert str(app.finish_button["state"]) == "disabled"
        for _ in range(4):
            app.record_button.invoke()
            app.stop_button.invoke()
            pump_until(lambda: not app.busy)
            app.confirmed.set(True)
            app.refresh_controls()
            app.save_button.invoke()
            pump_until(lambda: not app.busy)
        app.finish_button.invoke()
        pump_until(lambda: not app.busy)
        assert not errors
        assert notices
        assert app.session.draft.completed
        assert session.store.voice_path(session.alias).exists()
        app.alias.set("MY-VOICE")
        app.open_button.invoke()
        assert app.alias.get() == "my-voice"
        assert app.alias_entry["values"] == ("my-voice",)
        app.reading_picker.current(0)
        app.select_reading()
        assert "passage 1 of 6" in app.heading.get()
        assert str(app.record_button["state"]) == "normal"
    finally:
        app.stop_playback()
        root.destroy()
