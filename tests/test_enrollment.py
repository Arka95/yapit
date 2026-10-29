import json
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import tone

from yapit.enrollment import (
    EnrollmentSession,
    MicrophoneRecorder,
    Reading,
)
from yapit.media import SAMPLE_RATE, decode_audio, ffmpeg, write_wav
from yapit.storage import Workspace
from yapit.voices import load_voice


@pytest.fixture
def session(tmp_path, monkeypatch):
    readings = [
        Reading(title="First passage", text="Read this first short passage at a natural pace."),
        Reading(
            title="Second passage", text="Read this second passage with a clear and relaxed voice."
        ),
    ]
    monkeypatch.setattr("yapit.enrollment.READINGS", readings)
    return EnrollmentSession(Workspace(tmp_path / "workspace"), "my-voice", consent=True)


def take(session, seconds=4, frequency=200, volume=1):
    path = session.new_take()
    write_wav(path, tone(seconds, frequency) * volume)
    return path


def test_alias_consent_and_no_overwrite_guards(enrolled):
    with pytest.raises(ValueError, match="consent"):
        EnrollmentSession(enrolled, "new-alias", consent=False)
    with pytest.raises(ValueError, match="already enrolled"):
        EnrollmentSession(enrolled, "test-voice", consent=True)
    with pytest.raises(ValueError, match="Name"):
        EnrollmentSession(enrolled, "..\\escape", consent=True)


def test_saved_text_resume_and_final_enrollment(session):
    raw = take(session)
    original = raw.read_bytes()
    with pytest.raises(ValueError, match="confirm"):
        session.save_take(raw, confirmed=False, normalize=True)
    first = session.save_take(raw, confirmed=True, normalize=True)
    assert first["quality"]["reasons"] == []
    assert raw.read_bytes() == original
    resumed = EnrollmentSession(session.workspace, session.alias, consent=True)
    assert resumed.index == 1
    assert resumed.draft.samples[0].text == resumed.draft.readings[0].text
    assert len(json.loads((resumed.folder / "samples.json").read_text())["samples"]) == 1
    with pytest.raises(ValueError, match="every passage"):
        resumed.finish()
    resumed.save_take(take(resumed, frequency=260), confirmed=True, normalize=True)
    assert resumed.finished
    voice = resumed.finish()
    assert voice.name == "my-voice"
    assert len(voice.samples) == 2
    assert load_voice(session.workspace, voice.name).consent
    with pytest.raises(ValueError, match="already enrolled"):
        EnrollmentSession(session.workspace, voice.name, consent=True)


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

    monkeypatch.setattr("yapit.enrollment.subprocess.Popen", synthetic_capture)
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

    from yapit.desktop import EnrollmentApp

    monkeypatch.setattr("yapit.desktop.microphone_devices", lambda: ["Test microphone"])
    root = tk.Tk()
    root.withdraw()
    app = EnrollmentApp(root, session.workspace)
    errors = []
    playback = []
    notices = []
    monkeypatch.setattr(
        "yapit.desktop.messagebox.showerror",
        lambda *args, **kwargs: errors.append(args),
    )
    monkeypatch.setattr(
        "yapit.desktop.messagebox.showinfo",
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
        app.consent.set(True)
        app.open_button.invoke()
        assert "passage 1 of 2" in app.heading.get()
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
        assert "passage 2 of 2" in app.heading.get()
        assert app.take is None
        assert not app.confirmed.get()
        assert json.loads(app.session.path.read_text())["samples"][0]["text"] == (
            session.draft.readings[0].text
        )
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
        assert session.workspace.voice_path(session.alias).exists()
    finally:
        app.stop_playback()
        root.destroy()
