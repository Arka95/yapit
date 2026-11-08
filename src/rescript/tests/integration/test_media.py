import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from yapit.common_utils.mediautils import run_media
from yapit.rescript.audio import extract_audio
from yapit.rescript.tests.test_rescript import segments_at
from yapit.rescript.transcription import transcribe
from yapit.storage import DataStore
from yapit.yapper.script_planner import load_script


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
