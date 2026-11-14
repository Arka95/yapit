import json
import math
import wave
from array import array
from pathlib import Path

import pytest


def test_profiles_reports_empty_store(run_yapit):
    result = run_yapit("profiles")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_script_generation_requires_cloud_opt_in(run_yapit):
    result = run_yapit("generate-script", "A quiet weekend")
    assert result.returncode != 0
    assert "cloud" in (result.stdout + result.stderr).lower()


@pytest.mark.parametrize(
    "arguments",
    [
        ("generate-audio",),
        ("generate-reel", "--prompt", "A trip", "--audio-path", "speech.wav"),
    ],
)
def test_ambiguous_or_missing_inputs_fail_before_generation(run_yapit, arguments):
    result = run_yapit(*arguments)
    assert result.returncode != 0
    assert "exactly one" in result.stdout + result.stderr


def test_audio_extraction_preserves_supported_recording(run_yapit, tmp_path):
    source = tmp_path / "source.wav"
    rate = 16000
    samples = array(
        "h", (round(4000 * math.sin(2 * math.pi * 220 * n / rate)) for n in range(rate))
    )
    with wave.open(str(source), "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(rate)
        recording.writeframes(samples.tobytes())
    original = source.read_bytes()
    result = run_yapit("extract_audio", str(source))
    assert result.returncode == 0, result.stderr
    exported = Path(json.loads(result.stdout)["audio"])
    assert exported.is_relative_to(tmp_path / "data")
    assert exported.read_bytes() == original
    assert source.read_bytes() == original
