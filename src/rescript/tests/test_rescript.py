import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from yapit.common_utils.mediautils import run_media
from yapit.rescript.audio import extract_audio
from yapit.rescript.transcription import (
    Segment,
    WhisperTranscriber,
    compose_scripts,
    transcribe,
)
from yapit.storage import DataStore
from yapit.yapper.script_planner import load_script


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


@pytest.mark.parametrize("video", [False, True])
def test_real_media_decode_exports_loadable_scripts_and_cleans_temp(tmp_path, video):
    audio = tmp_path / "input.wav"
    sf.write(audio, np.zeros(16000 * 80, dtype=np.float32), 16000)
    source = audio
    if video:
        source = tmp_path / "input.mp4"
        run_media(
            [
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:r=1",
                "-i",
                str(audio),
                "-shortest",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                str(source),
            ]
        )
    seen = []

    def recognize(path):
        seen.append(path)
        assert sf.info(path).samplerate == 16000
        assert sf.info(path).channels == 1
        return segments_at(20, 40, 60, 80)

    result = transcribe(
        source,
        output=DataStore().transcripts,
        backend=SimpleNamespace(recognize=recognize),
    )
    assert result["status"] == "needs_review"
    assert Path(result["audio"]).is_relative_to(DataStore().audio)
    assert Path(result["audio"]).exists()
    assert not seen[0].exists()
    assert len(result["scripts"]) == 2
    for part in result["scripts"]:
        assert load_script(Path(part["script"])).word_count == 6
        assert part["suggested_seconds"] == 40
    sidecar = json.loads(Path(result["transcript"]).read_text())
    assert len(sidecar["segments"]) == 4
    repeated = transcribe(source, backend=SimpleNamespace(recognize=recognize))
    assert repeated["name"] == result["name"] + ".1"
    assert Path(repeated["scripts"][0]["script"]).parent == DataStore().transcripts
    with pytest.raises(ValueError, match="subfolders"):
        transcribe(source, output=DataStore().transcripts / "export")


def test_failure_cleans_temp_and_does_not_export(tmp_path):
    source = tmp_path / "silent.wav"
    sf.write(source, np.zeros(16000, dtype=np.float32), 16000)
    seen = []

    def recognize(path):
        seen.append(path)
        return []

    with pytest.raises(ValueError, match="No speech"):
        transcribe(
            source,
            output=DataStore().transcripts,
            backend=SimpleNamespace(recognize=recognize),
        )
    assert not seen[0].exists()
    assert not list(DataStore().transcripts.glob("*.transcript.json"))


def test_timestamp_beyond_audio_rejected(tmp_path):
    source = tmp_path / "input.wav"
    sf.write(source, np.zeros(16000, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="past the decoded"):
        transcribe(
            source,
            output=DataStore().transcripts,
            backend=SimpleNamespace(recognize=lambda _: segments_at(5)),
        )
    assert not list(DataStore().transcripts.glob("*.transcript.json"))


@pytest.mark.parametrize("suffix", [".wav", ".mp3", ".flac"])
def test_extract_copies_supported_audio_unchanged(tmp_path, monkeypatch, suffix):
    monkeypatch.chdir(tmp_path)
    wav = tmp_path / "SOURCE.wav"
    sf.write(wav, np.zeros(24000, dtype=np.float32), 24000)
    source = wav
    if suffix != ".wav":
        source = tmp_path / f"SOURCE{suffix}"
        run_media(["-y", "-i", str(wav), str(source)])
    result = extract_audio(source)
    output = Path(result["audio"])
    assert result["action"] == "copied"
    assert output.parent == tmp_path / "data" / "audio"
    assert output.name == f"arka.source{suffix}"
    assert output.read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        extract_audio(source)


def test_extract_converts_other_decodable_container(tmp_path):
    source = tmp_path / "audio.unknown"
    sf.write(source, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    result = extract_audio(source, output_dir=DataStore().audio / "out")
    output = Path(result["audio"])
    assert result["action"] == "converted"
    assert output.suffix == ".wav"
    assert sf.info(output).samplerate == 24000
    assert sf.info(output).channels == 1


def test_extract_detects_video_by_content_not_extension(tmp_path):
    video = tmp_path / "video.mp4"
    run_media(
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(video),
        ]
    )
    disguised = tmp_path / "video.mp3"
    video.rename(disguised)
    result = extract_audio(disguised, output_dir=DataStore().audio / "out")
    assert result["action"] == "converted"
    assert sf.info(result["audio"]).samplerate == 24000


@pytest.mark.parametrize("silent_video", [False, True])
def test_extract_reports_unsupported_or_missing_audio(tmp_path, silent_video):
    source = tmp_path / "bad.mp4"
    if silent_video:
        run_media(
            [
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:r=1",
                "-t",
                "1",
                "-an",
                "-c:v",
                "libx264",
                str(source),
            ]
        )
    else:
        source.write_bytes(b"This is not audio or video.")
    with pytest.raises(ValueError, match="unsupported/corrupt media or no audio stream"):
        extract_audio(source, output_dir=DataStore().audio / "out")
    assert not (DataStore().audio / "out").exists()


def test_extract_empty_audio_is_error(tmp_path):
    source = tmp_path / "empty.wav"
    sf.write(source, np.zeros(0, dtype=np.float32), 24000)
    with pytest.raises(ValueError, match="empty"):
        extract_audio(source, output_dir=DataStore().audio / "out")
    assert not (DataStore().audio / "out").exists()


def test_default_transcript_store_and_canonical_audio_reuse(tmp_path):
    data = DataStore()
    source = data.audio / "arka.input.wav"
    sf.write(source, np.zeros(16000 * 5, dtype=np.float32), 16000)
    result = transcribe(source, backend=SimpleNamespace(recognize=lambda _: segments_at(5)))
    assert Path(result["directory"]).is_relative_to(data.transcripts)
    assert Path(result["audio"]) == source
    assert not (data.audio / "imports").exists()
    assert extract_audio(source)["action"] == "already_stored"


def test_output_overrides_cannot_escape_data_buckets(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(16000, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="Output must stay inside"):
        extract_audio(source, output_dir=tmp_path / "outside")
    with pytest.raises(ValueError, match="Output must stay inside"):
        transcribe(source, output=DataStore().generated_audio / "wrong-bucket")
    assert not (tmp_path / "outside").exists()
