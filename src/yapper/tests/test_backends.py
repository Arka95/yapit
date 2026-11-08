import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from typer.testing import CliRunner

from yapit.cli import app
from yapit.models import Voice
from yapit.storage import DataStore, Workspace
from yapit.yapper import backends
from yapit.yapper.backends import BackendSelection, asset_stamps, create_backend, select_backend
from yapit.yapper.generation import generate_audio, render_audio
from yapit.yapper.pocket import PocketSpeech
from yapit.yapper.worker import SpeechEngine, WorkerSpeech, backend_for


@pytest.fixture
def voice(tmp_path):
    directory = tmp_path / "data" / "speaker" / "audio"
    directory.mkdir(parents=True)
    (directory / "reference.wav").write_bytes(b"reference")
    return (
        Voice(
            name="speaker",
            consent=True,
            reference_audio="reference.wav",
            reference_text="Reference words.",
            samples=[],
            words_per_minute=140,
        ),
        directory,
    )


class FakeBackend:
    def __init__(self, voice, directory, *, model=None, events=None, level=0.1):
        self.voice, self.directory, self.source = voice, directory, model
        self.events = events if events is not None else []
        self.level = level

    def load(self):
        self.events.append(("load", self.source))

    def close(self):
        self.events.append(("close", self.source))

    def cache_identity(self):
        return {
            "model": self.source,
            "assets": asset_stamps([self.directory / self.voice.reference_audio]),
        }

    def describe(self):
        return {"model": self.source, "device": "other-device", "voice_mode": "reference"}

    def synthesize(self, text):
        self.events.append(("speech", self.source))
        return np.full(24, self.level, dtype=np.float32)


def test_registered_factory_receives_its_model_without_qwen_assumptions(voice, monkeypatch):
    monkeypatch.setattr(backends, "FACTORIES", {})
    backends.register_backend("custom", FakeBackend)
    speaker, directory = voice
    backend = create_backend(speaker, directory, BackendSelection("custom", "my-model"))
    assert isinstance(backend, FakeBackend)
    assert backend.source == "my-model"


def test_installed_backend_plugin_is_loaded_by_name(voice, monkeypatch):
    entry = SimpleNamespace(name="plugin", load=Mock(return_value=FakeBackend))
    monkeypatch.setattr(backends.importlib.metadata, "entry_points", lambda **kwargs: [entry])
    backend = create_backend(*voice, BackendSelection("plugin", "plugin-model"))
    assert backend.source == "plugin-model"
    entry.load.assert_called_once()


def test_unknown_backend_has_an_actionable_error(voice):
    with pytest.raises(ValueError, match="install its yapit.tts plugin"):
        create_backend(*voice, BackendSelection("not-installed"))


def test_backend_switch_does_not_inherit_another_models_identifier():
    previous = BackendSelection("first", "first-model")
    assert select_backend("second", default=previous) == BackendSelection("second")
    assert select_backend(model="replacement", default=previous) == BackendSelection(
        "first", "replacement"
    )


@pytest.mark.parametrize("source", [None, "config.yaml"])
def test_pocket_reuses_conditioning_and_normalizes_native_audio(voice, monkeypatch, source):
    speaker, directory = voice
    speaker.checkpoint = str(directory.parent / "qwen-checkpoint")
    wave = Mock()
    wave.detach.return_value = wave
    wave.cpu.return_value = wave
    wave.numpy.return_value = np.full(48, 0.1, dtype=np.float32)
    model = SimpleNamespace(
        sample_rate=12000,
        get_state_for_audio_prompt=Mock(return_value={"voice": "cached"}),
        generate_audio=Mock(return_value=wave),
    )
    loader = Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules, "pocket_tts", SimpleNamespace(TTSModel=SimpleNamespace(load_model=loader))
    )
    resample = Mock(return_value=np.full(96, 0.1, dtype=np.float32))
    monkeypatch.setattr(backends, "resample", resample)
    backend = PocketSpeech(speaker, directory, **({"model": source} if source else {}))
    first = backend.synthesize("First.")
    backend.synthesize("Second.")
    assert first.shape == (96,)
    assert np.isfinite(first).all()
    loader.assert_called_once_with(
        **({"config": source} if source else {"language": "english_2026-10"})
    )
    model.get_state_for_audio_prompt.assert_called_once_with(str(directory / "reference.wav"))
    assert model.generate_audio.call_count == 2
    assert resample.call_args.args[1] == 12000
    assert backend.describe()["voice_mode"] == "reference"


def test_worker_reuses_same_model_and_separates_backend_and_model_caches(voice, monkeypatch):
    events = []
    monkeypatch.setitem(
        backends.FACTORIES,
        "first",
        lambda v, d, **kwargs: FakeBackend(v, d, events=events, **kwargs),
    )
    monkeypatch.setitem(
        backends.FACTORIES,
        "second",
        lambda v, d, **kwargs: FakeBackend(v, d, events=events, level=0.2, **kwargs),
    )
    engine = SpeechEngine(DataStore(), backend="first", model="one")
    first = engine.synthesize(*voice, "Same words.", True)
    assert np.array_equal(first, engine.synthesize(*voice, "Same words.", True))
    engine.synthesize(*voice, "Same words.", True, model="two")
    second = engine.synthesize(*voice, "Same words.", True, backend="second", model="two")
    assert not np.array_equal(first, second)
    assert [event for event in events if event[0] == "load"] == [
        ("load", "one"),
        ("load", "two"),
        ("load", "two"),
    ]
    assert sum(event[0] == "speech" for event in events) == 3
    assert sum(event[0] == "close" for event in events) == 2
    assert engine.status["backend"] == "second"
    assert engine.status["device"] == "other-device"


def test_writing_metadata_does_not_reload_a_generic_model(voice, monkeypatch):
    events = []
    monkeypatch.setitem(
        backends.FACTORIES,
        "custom",
        lambda v, d: FakeBackend(v, d, events=events),
    )
    engine = SpeechEngine(DataStore(), backend="custom")
    engine.synthesize(*voice, "Same.", True)
    voice[0].writing_style = "Different writing style"
    engine.synthesize(*voice, "Same.", True)
    assert events == [("load", None), ("speech", None)]


def test_failed_model_load_closes_candidate_and_keeps_worker_unready(voice, monkeypatch):
    candidate = FakeBackend(*voice)
    candidate.load = Mock(side_effect=RuntimeError("load failed"))
    candidate.close = Mock()
    monkeypatch.setitem(backends.FACTORIES, "failing", lambda *args: candidate)
    engine = SpeechEngine(DataStore(), backend="failing")
    with pytest.raises(RuntimeError, match="load failed"):
        engine.prewarm(*voice)
    candidate.close.assert_called_once()
    assert engine.backend is None and not engine.status["ready"]


def test_old_worker_cannot_silently_ignore_requested_backend(voice, monkeypatch):
    monkeypatch.setattr("yapit.yapper.worker.read_state", lambda data: {"port": 1})
    assert isinstance(backend_for(*voice), WorkerSpeech)
    with pytest.raises(RuntimeError, match="Restart"):
        backend_for(*voice, backend_name="pocket")


def test_worker_client_forwards_model_selection_without_loading_a_model(voice, monkeypatch):
    request = Mock(return_value=(np.full(24, 0.1, dtype="<f4").tobytes(), {}))
    monkeypatch.setattr("yapit.yapper.worker._request", request)
    client = WorkerSpeech(*voice, {"backend_selection": True}, backend="custom", model="tiny")
    assert client.synthesize("Hello.").shape == (24,)
    payload = request.call_args.args[2]
    assert (payload["backend"], payload["model"]) == ("custom", "tiny")


def test_audio_api_forwards_backend_and_model_without_media_execution(voice, tmp_path, monkeypatch):
    renderer = Mock(return_value=({"status": "complete"}, None))
    monkeypatch.setattr("yapit.yapper.generation.select_voice", lambda *args: voice)
    monkeypatch.setattr("yapit.yapper.generation.render_audio", renderer)
    generate_audio(
        "Hello.",
        workspace=Workspace(tmp_path / "workspace"),
        voice_name="speaker",
        backend_name="custom",
        model="tiny",
    )
    assert renderer.call_args.kwargs["backend_name"] == "custom"
    assert renderer.call_args.kwargs["model"] == "tiny"


def test_owned_backend_is_closed_when_synthesis_fails(voice, tmp_path, monkeypatch):
    from yapit.models import Script

    speaker, directory = voice
    backend = FakeBackend(speaker, directory)
    backend.synthesize = Mock(side_effect=RuntimeError("synthesis failed"))
    backend.close = Mock()
    monkeypatch.setattr("yapit.yapper.generation.backend_for", lambda *args, **kwargs: backend)
    job = Workspace(tmp_path / "workspace").output("failure", speaker.name)
    with pytest.raises(RuntimeError, match="synthesis failed"):
        render_audio(Script.from_text("Hello."), speaker, directory, job, seconds=5)
    backend.close.assert_called_once()


def test_qwen_explicit_base_model_overrides_profile_checkpoint(voice, monkeypatch):
    from yapit.yapper.speech import QwenSpeech

    speaker, directory = voice
    speaker.checkpoint = str(directory.parent / "trained")
    model = SimpleNamespace(
        model=SimpleNamespace(tts_model_type="base"),
        create_voice_clone_prompt=Mock(return_value="conditioning"),
        generate_voice_clone=Mock(return_value=([np.full(24, 0.1)], 24000)),
    )
    loader = Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(float32="fp32", cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=loader)),
    )
    monkeypatch.setenv("YAPIT_DEVICE", "cpu")
    backend = QwenSpeech(speaker, directory, model="Qwen/another-Base")
    assert backend.synthesize("Hello.").shape == (24,)
    assert loader.call_args.args[0] == "Qwen/another-Base"
    assert backend.describe()["voice_mode"] == "reference"
    model.create_voice_clone_prompt.assert_called_once()


def test_audiogen_cli_exposes_backend_and_model_without_model_execution(monkeypatch):
    generate = Mock(return_value={"status": "complete"})
    monkeypatch.setattr("yapit.yapper.cli.load_dotenv", lambda: None)
    monkeypatch.setattr("yapit.yapper.cli.generate_audio", generate)
    result = CliRunner().invoke(
        app, ["audiogen", "--text", "Hello.", "--backend", "custom", "--model", "tiny"]
    )
    assert result.exit_code == 0, result.output
    assert generate.call_args.kwargs["backend_name"] == "custom"
    assert generate.call_args.kwargs["model"] == "tiny"


def test_worker_cli_exposes_backend_and_model_without_model_execution(monkeypatch):
    run = Mock()
    monkeypatch.setattr("yapit.cli.data_store", lambda: "data")
    monkeypatch.setattr("yapit.yapper.worker.run_worker", run)
    result = CliRunner().invoke(
        app, ["voice-worker", "speaker", "--backend", "custom", "--model", "tiny"]
    )
    assert result.exit_code == 0, result.output
    run.assert_called_once_with("data", "speaker", backend="custom", model="tiny")
