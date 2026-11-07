import numpy as np
import pytest

from yapit.common_utils.fileutils import save_json
from yapit.common_utils.mediautils import SAMPLE_RATE, write_wav
from yapit.enrollment import enroll
from yapit.models import Beat, Script
from yapit.storage import Workspace


@pytest.fixture(autouse=True)
def isolated_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("YAPIT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("YAPIT_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("YAPIT_MODELS_DIR", str(tmp_path / "models"))


def tone(seconds, frequency=180):
    time = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * time) ** 2
    return (0.15 * envelope * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


class ToneSpeech:
    def __init__(self, seconds=3.1):
        self.seconds = seconds
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        return tone(self.seconds, 180 + len(self.calls) * 40)


def script():
    return Script(
        title="Weekend",
        beats=[
            Beat(text="Choose a nearby base for the weekend.", pause_ms=200),
            Beat(text="Keep one afternoon free, then enjoy the slower pace.", pause_ms=300),
        ],
    )


@pytest.fixture
def enrolled(tmp_path):
    workspace = Workspace(tmp_path / "workspace")
    samples = []
    for index in range(5):
        path = tmp_path / f"sample-{index}.wav"
        write_wav(path, tone(4, 180 + 30 * index))
        samples.append({"audio": str(path), "text": f"This is clear test passage {index}."})
    manifest = tmp_path / "samples.json"
    save_json(manifest, {"samples": samples})
    enroll(workspace.data, "test-voice", manifest)
    return workspace
