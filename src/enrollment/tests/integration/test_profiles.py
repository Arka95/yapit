import json

import pytest
from pydantic import ValidationError

from yapit.common_utils.fileutils import save_json
from yapit.enrollment import enroll, load_voice


def test_reference_reimport_preserves_writing_metadata_and_original_audio(enrolled, tmp_path):
    voice = load_voice(enrolled, "test-voice")
    voice.writing_style = "relaxed"
    voice.frequent_words = ["lovely"]
    voice.travel_preferences = {"pace": "slow"}
    save_json(enrolled.voice_path(voice.name), voice)
    source = enrolled.voice_audio_dir(voice.name) / voice.reference_audio
    before = source.read_bytes()
    manifest = tmp_path / "reference.json"
    save_json(manifest, {"samples": [{"audio": str(source), "text": voice.reference_text}]})
    saved = enroll(enrolled, voice.name, manifest)
    assert len(saved.samples) == 1
    assert saved.writing_style == "relaxed"
    assert saved.frequent_words == ["lovely"]
    assert saved.travel_preferences == {"pace": "slow"}
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("samples", "not a list"),
        ("words_per_minute", float("nan")),
        ("frequent_words", "insane"),
        ("preferred_greeting", ""),
    ],
)
def test_flexible_metadata_preserves_declared_field_validation(enrolled, field, value):
    path = enrolled.voice_path("test-voice")
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_voice(enrolled, "test-voice")
