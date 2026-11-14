import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from yapit.rescript.audio import extract_audio
from yapit.rescript.transcription import (
    Segment,
    WhisperTranscriber,
    compose_scripts,
    extract_script,
    timeline_from_segments,
    transcribe,
)
from yapit.storage import DataStore


def segments_at(*ends):
    start = 0
    result = []
    for index, end in enumerate(ends):
        result.append(Segment(start=start, end=end, text=f"Sentence number {index}."))
        start = end
    return result


def test_whisper_mode_environment_and_explicit_overrides(monkeypatch):
    monkeypatch.setenv("YAPIT_ASR_DEVICE", "cuda")
    monkeypatch.setenv("YAPIT_ASR_COMPUTE_TYPE", "float16")
    selected = WhisperTranscriber()
    assert (selected.device, selected.compute_type) == ("cuda", "float16")
    explicit = WhisperTranscriber(device="cpu", compute_type="int8")
    assert (explicit.device, explicit.compute_type) == ("cpu", "int8")
    device_only = WhisperTranscriber(device="cpu")
    assert (device_only.device, device_only.compute_type) == ("cpu", "int8")


def test_whisper_defaults_remain_cpu_without_mode(monkeypatch):
    monkeypatch.delenv("YAPIT_ASR_DEVICE", raising=False)
    monkeypatch.delenv("YAPIT_ASR_COMPUTE_TYPE", raising=False)
    monkeypatch.setenv("YAPIT_DEVICE", "cuda")
    selected = WhisperTranscriber()
    assert (selected.device, selected.compute_type) == ("cpu", "int8")


def test_partition_avoids_short_tail_and_preserves_text():
    segments = segments_at(20, 40, 60, 80)
    parts = compose_scripts(segments, "Trip")
    assert [(start, end) for _, start, end in parts] == [(0, 40), (40, 80)]
    assert [beat.text for script, _, _ in parts for beat in script.beats] == [
        segment.text for segment in segments
    ]


@pytest.mark.parametrize("duration", [5, 30, 60])
def test_short_unsplit_recordings(duration):
    parts = compose_scripts(segments_at(duration), "Short")
    assert len(parts) == 1
    assert parts[0][2] == duration


@pytest.mark.parametrize("ends", [(30, 61), (60, 61), (30, 60, 90)])
def test_impossible_split_does_not_accept_short_tail(ends):
    with pytest.raises(ValueError, match="Cannot form"):
        compose_scripts(segments_at(*ends), "Cannot split")


def test_sentence_boundary_required():
    segments = segments_at(40, 80)
    segments[0].text = "An unfinished thought"
    with pytest.raises(ValueError, match="sentence boundaries"):
        compose_scripts(segments, "No cut")


def test_pauses_capped_and_outer_silence_excluded():
    parts = compose_scripts(
        [
            Segment(start=10, end=20, text="First sentence."),
            Segment(start=22, end=30, text="Last sentence."),
        ],
        "Pauses",
    )
    script, start, end = parts[0]
    assert (start, end) == (10, 30)
    assert [beat.pause_ms for beat in script.beats] == [1200, 0]


def test_empty_and_invalid_order():
    with pytest.raises(ValueError, match="No speech"):
        compose_scripts([], "Silent")
    with pytest.raises(ValueError, match="ordered"):
        compose_scripts(
            [
                Segment(start=0, end=10, text="First."),
                Segment(start=9, end=20, text="Overlapping."),
            ],
            "Invalid",
        )


def test_schema_limits_prevent_export():
    with pytest.raises(ValueError, match="650 words"):
        compose_scripts(
            [Segment(start=0, end=60, text=("word " * 651).strip() + ".")],
            "Too many words",
        )


@pytest.mark.parametrize(
    "settings,local_only", [({}, True), ({"offline": True}, True), ({"offline": False}, False)]
)
def test_whisper_adapter_consumes_lazy_words_and_uses_local_settings(
    monkeypatch, tmp_path, settings, local_only
):
    calls = []
    words = [
        SimpleNamespace(start=0, end=1, word=" Hello"),
        SimpleNamespace(start=1, end=2, word=" world."),
        SimpleNamespace(start=3, end=4, word=" Goodbye."),
        SimpleNamespace(start=5, end=6, word=" unfinished"),
    ]

    def recognize(audio, **kwargs):
        calls.append(("transcribe", kwargs))

        def generate():
            calls.append(("consumed", audio))
            yield SimpleNamespace(text="Hello world. Goodbye. unfinished", words=words)

        return generate(), SimpleNamespace()

    def model(name, **kwargs):
        calls.append(("model", (name, kwargs)))
        return SimpleNamespace(transcribe=recognize)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=model))
    result = WhisperTranscriber(**settings).recognize(tmp_path / "audio.wav")
    assert [segment.text for segment in result] == ["Hello world.", "Goodbye.", "unfinished"]
    assert calls[0] == (
        "model",
        (
            "small.en",
            {
                "device": "cpu",
                "compute_type": "int8",
                "local_files_only": local_only,
                "download_root": str(tmp_path / "models" / "hub"),
            },
        ),
    )
    assert calls[1][1]["word_timestamps"] is True
    assert calls[1][1]["vad_filter"] is True
    assert calls[1][1]["task"] == "transcribe"
    assert calls[2][0] == "consumed"


def test_output_overrides_cannot_escape_data_buckets(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(16000, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="Output must stay inside"):
        extract_audio(source, output_dir=tmp_path / "outside")
    with pytest.raises(ValueError, match="Output must stay inside"):
        transcribe(source, output=DataStore().generated_audio / "wrong-bucket")
    assert not (tmp_path / "outside").exists()


def test_extraction_includes_each_script_json_without_rewriting(tmp_path, monkeypatch):
    from yapit.common_utils.fileutils import save_json
    from yapit.models import Script

    script = Script.from_text("First thought. Last thought.")
    path = tmp_path / "source.script.json"
    save_json(path, script)
    monkeypatch.setattr(
        "yapit.rescript.transcription.transcribe",
        lambda *args, **kwargs: {
            "status": "needs_review",
            "scripts": [{"script": str(path), "source_start": 2, "source_end": 10}],
        },
    )
    result = extract_script(tmp_path / "source.wav")
    assert result["status"] == "needs_review"
    assert result["scripts"][0]["content"] == script.model_dump(mode="json")
    assert result["scripts"][0]["source_start"] == 2


@pytest.mark.parametrize(
    "segments",
    [
        [],
        [Segment(start=0, end=4, text="First."), Segment(start=3, end=5, text="Overlap.")],
        [Segment(start=1, end=12, text="Past the end.")],
    ],
)
def test_audio_reel_timeline_rejects_missing_or_invalid_timing(segments):
    with pytest.raises(ValueError, match="No speech|timestamps"):
        timeline_from_segments(segments, 10)


def test_audio_reel_duration_limit_rejects_long_input_before_recognition(tmp_path, monkeypatch):
    from unittest.mock import Mock

    source = tmp_path / "source.wav"
    source.touch()
    monkeypatch.setattr(
        "yapit.rescript.transcription.extract_audio_stream",
        lambda *args, **kwargs: SimpleNamespace(stderr=b""),
    )
    monkeypatch.setattr(
        "yapit.rescript.transcription.sf.info", lambda *args: SimpleNamespace(duration=181)
    )
    recognize = Mock()
    with pytest.raises(ValueError, match="at most 180 seconds"):
        transcribe(source, max_seconds=180, backend=SimpleNamespace(recognize=recognize))
    recognize.assert_not_called()
