import numpy as np
import pytest

from yapit.common_utils.fileutils import save_json
from yapit.models import Sample, SampleManifest, Voice
from yapit.storage import Workspace
from yapit.training.dataset import training_manifest


@pytest.fixture(autouse=True)
def isolated_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("YAPIT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("YAPIT_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("YAPIT_MODELS_DIR", str(tmp_path / "models"))


@pytest.fixture
def training_workspace(tmp_path, monkeypatch):
    def write(path, data):
        path.write_bytes(np.asarray(data, dtype=np.float32).tobytes())

    def decode(path, *, max_seconds):
        data = np.frombuffer(path.read_bytes(), dtype=np.float32).copy()
        if len(data) > max_seconds * 1000:
            raise ValueError("Recording is too long.")
        return data

    for module in ("yapit.training.dataset", "yapit.enrollment.session"):
        monkeypatch.setattr(f"{module}.write_wav", write)
        monkeypatch.setattr(f"{module}.decode_audio", decode)
    for module in (
        "yapit.enrollment.profiles",
        "yapit.enrollment.session",
        "yapit.training.trainer",
    ):
        monkeypatch.setattr(f"{module}.SAMPLE_RATE", 1000)
    monkeypatch.setattr("yapit.training.trainer.decode_audio", decode)
    workspace = Workspace(tmp_path / "workspace")
    folder = workspace.voice_audio_dir("test-voice")
    folder.mkdir(parents=True)
    reference = folder / "reference.wav"
    write(reference, np.full(60000, 0.1))
    voice = Voice(
        name="test-voice",
        consent=True,
        reference_audio=reference.name,
        reference_text="Enrollment reference.",
        samples=[Sample(audio=reference.name, text="Enrollment reference.")],
        words_per_minute=150,
    )
    save_json(workspace.voice_path(voice.name), voice)
    samples = []
    for index in range(16):
        path = folder / f"sample-{index}.wav"
        write(path, np.full(6000, 0.05 + index / 1000))
        samples.append(Sample(audio=str(path), text=f"This is clear test passage {index}."))
    save_json(training_manifest(workspace.data, voice.name), SampleManifest(samples=samples))
    return workspace
