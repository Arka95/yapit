import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from typer.testing import CliRunner

from yapit.enrollment import EnrollmentSession, audio_quality, enroll, load_readings, profile_names
from yapit.enrollment.cli import app
from yapit.enrollment.profiles import load_voice
from yapit.enrollment.tests.conftest import tone
from yapit.media import SAMPLE_RATE, write_wav
from yapit.storage import DataStore, data_filename, save_json


def test_profile_merge_replaces_same_filename_preserves_settings_and_lowercases(enrolled, tmp_path):
    voice = load_voice(enrolled, "test-voice")
    voice.checkpoint = "keep-this-checkpoint"
    voice.speaker = "test-voice"
    save_json(enrolled.voice_path(voice.name), voice)
    source = tmp_path / "TAKE-1.WAV"
    write_wav(source, tone(frequency=390))
    manifest = tmp_path / "update.json"
    save_json(manifest, {"samples": [{"audio": str(source), "text": "The replacement text."}]})
    unchanged = enrolled.voice_audio_dir(voice.name) / voice.samples[1].audio
    old = unchanged.read_bytes()
    result = enroll(enrolled, "TEST-VOICE", manifest)
    assert len(result.samples) == 5
    assert result.checkpoint == voice.checkpoint
    assert result.speaker == voice.speaker
    assert unchanged.read_bytes() == old
    changed = next(sample for sample in result.samples if sample.audio == "test-voice.take-1.wav")
    assert changed.text == "The replacement text."
    journal = json.loads((enrolled.root / "test-voice" / "test-voice.journal.json").read_text())
    assert journal["sample_count"] == 5
    row = next(row for row in journal["samples"] if row["audio"].endswith(changed.audio))
    assert (
        row["sha256"]
        == hashlib.sha256(
            (enrolled.voice_audio_dir(voice.name) / changed.audio).read_bytes()
        ).hexdigest()
    )
    assert profile_names(enrolled) == ["test-voice"]
    assert all(
        path.name == path.name.lower() and path.name.startswith("test-voice.")
        for path in (enrolled.root / "test-voice").rglob("*")
        if path.is_file()
    )


def test_new_profile_minimum_and_invalid_update_do_not_modify_existing(enrolled, tmp_path):
    source = tmp_path / "short-set.wav"
    write_wav(source, tone(frequency=400))
    manifest = tmp_path / "samples.json"
    save_json(manifest, {"samples": [{"audio": str(source), "text": "Just one passage."}]})
    with pytest.raises(ValueError, match="at least 5"):
        enroll(enrolled, "new-profile", manifest)
    assert not enrolled.voice_path("new-profile").exists()
    original = enrolled.voice_path("test-voice").read_bytes()
    copy = tmp_path / "other-name.wav"
    shutil.copyfile(enrolled.voice_audio_dir("test-voice") / "test-voice.take-1.wav", copy)
    save_json(manifest, {"samples": [{"audio": str(copy), "text": "Duplicate audio."}]})
    with pytest.raises(ValueError, match="Duplicate"):
        enroll(enrolled, "test-voice", manifest)
    assert enrolled.voice_path("test-voice").read_bytes() == original
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"not audio")
    save_json(manifest, {"samples": [{"audio": str(outside), "text": "An outside path."}]})
    with pytest.raises(ValueError, match="inside"):
        enroll(enrolled, "outside", manifest, confined=True)


def test_recorded_sample_replacement_reuses_filename_and_updates_journal(enrolled):
    session = EnrollmentSession(enrolled, "Another")
    for index in range(5):
        source = session.new_take()
        write_wav(source, tone(frequency=200 + 30 * index))
        session.save_take(source, True, True)
    session.finish()
    reopened = EnrollmentSession(enrolled, "ANOTHER")
    reopened.select_sample(0)
    target = reopened.audio_folder / "another.audio_sample_1.wav"
    before = target.read_bytes()
    source = reopened.new_take()
    write_wav(source, tone(frequency=420))
    reopened.save_take(source, True, True)
    assert target.read_bytes() != before
    assert len(reopened.samples) == 5
    reopened.finish()
    assert profile_names(enrolled) == ["another", "test-voice"]
    assert len(load_voice(enrolled, "another").samples) == 5
    assert not source.exists()
    assert not reopened.path.exists()
    assert target.exists()


def test_profiles_are_read_only_from_data(tmp_path, monkeypatch):
    old = tmp_path / "workspace" / "voices" / "arka"
    old.mkdir(parents=True)
    save_json(old / "profile.json", {"unused": "not a current profile"})
    monkeypatch.setenv("YAPIT_WORKSPACE", str(tmp_path / "workspace"))
    store = DataStore(tmp_path / "data")
    assert profile_names(store) == []
    with pytest.raises(FileNotFoundError):
        load_voice(store, "arka")
    assert store.voice_path("arka") == tmp_path / "data" / "arka" / "arka.profile.json"


def test_static_scripts_are_shared_and_sorted_numerically(tmp_path):
    readings = load_readings()
    assert readings[9].sample_id == "sample_10"
    store = DataStore(tmp_path / "data")
    first, second = EnrollmentSession(store, "first"), EnrollmentSession(store, "second")
    assert first.draft.readings == second.draft.readings == readings
    assert not list(store.root.rglob("sampleScripts"))
    assert json.loads(first.path.read_text())["readings"][0]["text"] == readings[0].text


@pytest.mark.parametrize("data", [np.zeros(4 * SAMPLE_RATE), np.ones(4 * SAMPLE_RATE)])
def test_signal_checks_reject_silence_and_clipping(data):
    assert audio_quality(data)["reasons"]


def test_module_cli_works_without_model_imports_or_network(tmp_path):
    script = """
import importlib.abc
import socket
import sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'transformers', 'qwen_tts', 'faster_whisper', 'openai', 'huggingface_hub'
        }:
            raise AssertionError('Model dependency accessed: ' + fullname)
sys.meta_path.insert(0, NoModels())
socket.create_connection = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError('Network accessed')
)
from yapit.enrollment import EnrollmentSession
from yapit.storage import DataStore
session = EnrollmentSession(DataStore(sys.argv[1]), 'ARKA')
assert session.alias == 'arka'
assert not session.ready
import numpy as np
from yapit.media import SAMPLE_RATE, write_wav
for index in range(5):
    path = session.new_take()
    t = np.arange(4 * SAMPLE_RATE) / SAMPLE_RATE
    write_wav(path, (0.15 * np.sin(2 * np.pi * (180 + index * 30) * t)).astype(np.float32))
    session.save_take(path, True, False)
assert len(session.finish().samples) == 5
from yapit.enrollment.cli import app
from typer.testing import CliRunner
result = CliRunner().invoke(app, ['profiles'])
assert result.exit_code == 0, result.output
from yapit.cli import app as main_app
from pathlib import Path
result = CliRunner().invoke(main_app, [
    'voice-enroll', 'arka', '--manifest', str(session.folder / 'arka.samples.json')
])
assert result.exit_code == 0, result.output
assert not (Path(sys.argv[2]) / 'enrollment' / 'arka').exists()
assert not list(session.store.root.rglob('*.log'))
assert not list(session.store.root.rglob('*.raw.*'))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "data"), str(tmp_path / "workspace")],
        env={**os.environ, "YAPIT_WORKSPACE": str(tmp_path / "workspace")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    output = CliRunner().invoke(app, ["profiles"])
    assert output.exit_code == 0


def test_unified_data_name_handles_case_prefix_and_path_like_input():
    assert data_filename("ARKA.My Travel", "ARKA", ".WAV") == "arka.my_travel.wav"
    assert data_filename("..\\TRIP", "ARKA", ".SCRIPT.JSON") == "arka.trip.script.json"
    with pytest.raises(ValueError, match="extension"):
        data_filename("trip", "arka", ".wav\\escape")


def test_same_normalized_filename_in_manifest_uses_last_recording(enrolled, tmp_path):
    first = tmp_path / "first" / "TAKE-1.WAV"
    last = tmp_path / "last" / "take-1.wav"
    first.parent.mkdir()
    last.parent.mkdir()
    write_wav(first, tone(frequency=380))
    write_wav(last, tone(frequency=410))
    manifest = tmp_path / "merge.json"
    save_json(
        manifest,
        {
            "samples": [
                {"audio": str(first), "text": "First replacement."},
                {"audio": str(last), "text": "Last replacement."},
            ]
        },
    )
    voice = enroll(enrolled, "TEST-VOICE", manifest)
    assert len(voice.samples) == 5
    sample = next(sample for sample in voice.samples if sample.audio == "test-voice.take-1.wav")
    assert sample.text == "Last replacement."
