import numpy as np
import pytest

from yapit.enrollment import enroll
from yapit.media import SAMPLE_RATE, write_wav
from yapit.storage import Workspace, save_json


@pytest.fixture(autouse=True)
def isolated_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("YAPIT_DATA_DIR", str(tmp_path / "data"))


def tone(seconds=4, frequency=180):
    time = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * time) ** 2
    return (0.15 * envelope * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


@pytest.fixture
def enrolled(tmp_path):
    workspace = Workspace(tmp_path / "workspace")
    samples = []
    for index in range(5):
        path = tmp_path / f"take-{index + 1}.wav"
        write_wav(path, tone(frequency=180 + 30 * index))
        samples.append({"audio": str(path), "text": f"This is clear test passage {index + 1}."})
    manifest = tmp_path / "samples.json"
    save_json(manifest, {"samples": samples})
    enroll(workspace, "test-voice", manifest)
    return workspace
