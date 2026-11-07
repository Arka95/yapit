from pathlib import Path

import numpy as np
import pytest

from yapit.common_utils.fileutils import save_json
from yapit.common_utils.mediautils import SAMPLE_RATE, write_wav
from yapit.storage import Workspace
from yapit.voices import enroll


@pytest.fixture(autouse=True)
def isolated_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("YAPIT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("YAPIT_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("YAPIT_MODELS_DIR", str(tmp_path / "models"))


def tone(seconds: float, frequency: float = 180) -> np.ndarray:
    time = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * time) ** 2
    return (0.15 * envelope * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


@pytest.fixture
def enrolled(tmp_path: Path) -> Workspace:
    workspace = Workspace(tmp_path / "workspace")
    incoming = workspace.data.audio / "incoming"
    incoming.mkdir(parents=True)
    write_wav(incoming / "one.wav", tone(4))
    write_wav(incoming / "two.wav", tone(5, 210))
    extra = []
    for index in range(3):
        path = incoming / f"extra-{index}.wav"
        write_wav(path, tone(4, 240 + index * 30))
        extra.append({"audio": str(path), "text": f"A clear additional passage {index}."})
    manifest = workspace.data.transcripts / "incoming" / "samples.json"
    save_json(
        manifest,
        {
            "samples": [
                {
                    "audio": str(incoming / "one.wav"),
                    "text": "A calm first sentence to explain the weekend plan.",
                },
                {
                    "audio": str(incoming / "two.wav"),
                    "text": "A second sentence with a slightly different expressive delivery.",
                },
                *extra,
            ]
        },
    )
    enroll(workspace, "test-voice", manifest, consent=True)
    return workspace
