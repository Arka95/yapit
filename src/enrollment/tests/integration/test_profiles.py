import hashlib
import json
import shutil

import pytest
from pydantic import ValidationError

from yapit.common_utils.fileutils import save_json
from yapit.common_utils.mediautils import write_wav
from yapit.enrollment import EnrollmentSession, enroll, load_readings, profile_names
from yapit.enrollment.profiles import load_voice
from yapit.enrollment.tests.integration.conftest import tone
from yapit.storage import DataStore


def test_profile_merge_replaces_same_filename_preserves_settings_and_lowercases(enrolled, tmp_path):
    voice = load_voice(enrolled, "test-voice")
    voice.checkpoint = "keep-this-checkpoint"
    voice.speaker = "test-voice"
    voice.writing_style = "relaxed and informal"
    voice.frequent_words = ["insane", "so cool"]
    voice.preferred_greeting = "yo guys"
    voice.travel_preferences = {"pace": "slow"}
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
    saved = load_voice(enrolled, "test-voice")
    assert saved.writing_style == voice.writing_style
    assert saved.frequent_words == voice.frequent_words
    assert saved.preferred_greeting == voice.preferred_greeting
    assert saved.model_dump()["travel_preferences"] == {"pace": "slow"}
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


def test_existing_profile_gains_defaults_and_preserves_extra_metadata(enrolled):
    path = enrolled.voice_path("test-voice")
    payload = json.loads(path.read_text())
    for field in ("writing_style", "frequent_words", "preferred_greeting"):
        payload.pop(field, None)
    payload["custom_preferences"] = {"topics": ["travel"], "emoji": False}
    save_json(path, payload)
    profile = load_voice(enrolled, "test-voice")
    assert profile.writing_style == "energetic and informal"
    assert profile.frequent_words == []
    assert profile.preferred_greeting is None
    save_json(path, profile)
    saved = json.loads(path.read_text())
    assert saved["custom_preferences"] == payload["custom_preferences"]
    assert saved["writing_style"] == "energetic and informal"


@pytest.mark.parametrize(
    "field,value",
    [
        ("samples", "not a list"),
        ("words_per_minute", float("nan")),
        ("frequent_words", "insane"),
        ("preferred_greeting", ""),
    ],
)
def test_flexible_metadata_does_not_disable_declared_field_validation(enrolled, field, value):
    path = enrolled.voice_path("test-voice")
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_voice(enrolled, "test-voice")


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
    old = tmp_path / "workspace" / "voices" / "testuser"
    old.mkdir(parents=True)
    save_json(old / "profile.json", {"unused": "not a current profile"})
    monkeypatch.setenv("YAPIT_WORKSPACE", str(tmp_path / "workspace"))
    store = DataStore(tmp_path / "data")
    assert profile_names(store) == []
    with pytest.raises(FileNotFoundError):
        load_voice(store, "testuser")
    assert store.voice_path("testuser") == tmp_path / "data" / "testuser" / "testuser.profile.json"


def test_static_scripts_are_shared_and_sorted_numerically(tmp_path):
    readings = load_readings()
    assert readings[9].sample_id == "sample_10"
    store = DataStore(tmp_path / "data")
    first, second = EnrollmentSession(store, "first"), EnrollmentSession(store, "second")
    assert first.draft.readings == second.draft.readings == readings
    assert not list(store.root.rglob("sampleScripts"))
    assert json.loads(first.path.read_text())["readings"][0]["text"] == readings[0].text


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
