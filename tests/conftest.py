from pathlib import Path

import numpy as np
import pytest

from yapit.media import SAMPLE_RATE, write_wav
from yapit.storage import Workspace, save_json
from yapit.voices import enroll


def tone(seconds: float, frequency: float = 180) -> np.ndarray:
    time = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * time) ** 2
    return (0.15 * envelope * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


@pytest.fixture
def enrolled(tmp_path: Path) -> Workspace:
    workspace = Workspace(tmp_path / "workspace")
    incoming = workspace.root / "incoming"
    incoming.mkdir()
    write_wav(incoming / "one.wav", tone(4))
    write_wav(incoming / "two.wav", tone(5, 210))
    manifest = incoming / "samples.json"
    save_json(
        manifest,
        {
            "samples": [
                {"audio": "one.wav", "text": "A calm first sentence to explain the weekend plan."},
                {
                    "audio": "two.wav",
                    "text": "A second sentence with a slightly different expressive delivery.",
                },
            ]
        },
    )
    enroll(workspace, "test-voice", manifest, consent=True)
    return workspace
