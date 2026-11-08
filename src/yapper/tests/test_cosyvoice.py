import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from yapit.models import Voice
from yapit.yapper import backends, isolated, isolated_runtime
from yapit.yapper.backends import BackendSelection, create_backend
from yapit.yapper.cosyvoice import DEFAULT_MODEL, CosyVoiceSpeech
from yapit.yapper.cosyvoice_runtime import CosyVoiceSession


@pytest.fixture
def setup(tmp_path, monkeypatch):
    audio = tmp_path / "data" / "speaker" / "audio"
    audio.mkdir(parents=True)
    sf.write(audio / "reference.wav", np.full(24, 0.1), 24000)
    voice = Voice(
        name="speaker",
        consent=True,
        reference_audio="reference.wav",
        reference_text="Exact reference words.",
        samples=[],
        words_per_minute=140,
        checkpoint=str(tmp_path / "qwen-checkpoint"),
    )
    models = tmp_path / "models"
    model = models / "cosyvoice2-0.5b"
    model.mkdir(parents=True)
    (model / "cosyvoice2.yaml").write_text("model")
    python = tmp_path / "runtime" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"runtime")
    source = models / "cosyvoice_source"
    cli = source / "cosyvoice" / "cli"
    cli.mkdir(parents=True)
    (cli / "cosyvoice.py").write_text("sdk")
    monkeypatch.setenv("YAPIT_MODELS_DIR", str(models))
    monkeypatch.setenv("YAPIT_COSYVOICE_PYTHON", str(python))
    monkeypatch.setenv("YAPIT_DEVICE", "cpu")
    return voice, audio, model


def test_generic_registry_selects_cosyvoice_without_inheriting_qwen_weights(setup):
    voice, audio, model = setup
    backend = create_backend(voice, audio, BackendSelection("cosyvoice", DEFAULT_MODEL))
    assert isinstance(backend, CosyVoiceSpeech)
    assert backend.model_dir == model
    assert backend.describe()["voice_mode"] == "reference"
    assert "qwen-checkpoint" not in json.dumps(backend.cache_identity())


def test_adapter_reuses_runtime_normalizes_audio_and_cleans_owned_files(setup, monkeypatch):
    voice, audio, _ = setup
    runtime = Mock()

    def request(payload, **kwargs):
        if payload["action"] == "load":
            return {"device": "cpu", "dtype": "float32"}
        sf.write(payload["output"], np.full(12, 0.1), 12000, subtype="FLOAT")
        return {"sample_rate": 12000}

    runtime.request.side_effect = request
    factory = Mock(return_value=runtime)
    monkeypatch.setattr("yapit.yapper.cosyvoice.RuntimeProcess", factory)
    monkeypatch.setattr(backends, "resample", lambda wave, rate: np.repeat(wave, 2))
    backend = CosyVoiceSpeech(voice, audio)
    first = backend.synthesize("First words.")
    root = Path(backend.temporary.name)
    backend.synthesize("Second words.")
    assert first.shape == (24,) and first.dtype == np.float32
    assert not (root / "speech.wav").exists()
    factory.assert_called_once()
    load = runtime.request.call_args_list[0].args[0]
    assert load["reference_text"] == voice.reference_text
    assert load["reference"] == str(audio / voice.reference_audio)
    backend.close()
    backend.close()
    runtime.close.assert_called_once()
    assert not root.exists()


def test_failed_runtime_load_closes_child_and_removes_private_directory(setup, monkeypatch):
    runtime = Mock()
    runtime.request.side_effect = RuntimeError("missing SDK dependency")
    monkeypatch.setattr("yapit.yapper.cosyvoice.RuntimeProcess", lambda *args, **kwargs: runtime)
    backend = CosyVoiceSpeech(*setup[:2])
    with pytest.raises(RuntimeError, match="missing SDK"):
        backend.load()
    runtime.close.assert_called_once()
    assert backend.runtime is None and backend.temporary is None


def test_failed_synthesis_releases_runtime_and_allows_a_fresh_load(setup, monkeypatch):
    runtime = Mock()
    runtime.request.side_effect = [{"device": "cpu"}, RuntimeError("runtime crashed")]
    factory = Mock(return_value=runtime)
    monkeypatch.setattr("yapit.yapper.cosyvoice.RuntimeProcess", factory)
    backend = CosyVoiceSpeech(*setup[:2])
    with pytest.raises(RuntimeError, match="runtime crashed"):
        backend.synthesize("Approved words.")
    runtime.close.assert_called_once()
    assert backend.runtime is None and backend.temporary is None
    runtime.request.side_effect = [{"device": "cpu"}]
    backend.load()
    assert factory.call_count == 2
    backend.close()


def test_missing_runtime_is_actionable_and_does_not_spawn(setup, monkeypatch):
    spawn = Mock()
    monkeypatch.setattr("yapit.yapper.cosyvoice.RuntimeProcess", spawn)
    monkeypatch.setenv("YAPIT_COSYVOICE_PYTHON", str(setup[1] / "missing.exe"))
    with pytest.raises(RuntimeError, match="YAPIT_COSYVOICE_PYTHON"):
        CosyVoiceSpeech(*setup[:2]).load()
    spawn.assert_not_called()


def test_cosyvoice_rejects_reference_escape(setup):
    voice, audio, _ = setup
    voice.reference_audio = "../../other.wav"
    with pytest.raises(ValueError, match="inside"):
        CosyVoiceSpeech(voice, audio)


def test_model_weight_change_invalidates_adapter_identity(setup):
    weights = setup[2] / "llm.pt"
    weights.write_bytes(b"first")
    backend = CosyVoiceSpeech(*setup[:2])
    before = backend.cache_identity()
    weights.write_bytes(b"different weights")
    assert before != backend.cache_identity()


@pytest.mark.parametrize(
    "response",
    ['{"ok": false, "error": "SDK failed"}\n', "not json\n", ""],
)
def test_runtime_failure_reaps_owned_process(response, monkeypatch, tmp_path):
    process = Mock()
    process.stdin = io.StringIO()
    process.stdout = io.StringIO(response)
    process.poll.return_value = None
    monkeypatch.setattr(isolated.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        isolated.threading,
        "Thread",
        lambda *, target, **kwargs: SimpleNamespace(start=target),
    )
    runtime = isolated.RuntimeProcess(["python", "handler.py"], cwd=tmp_path, env={})
    with pytest.raises((RuntimeError, ValueError)):
        runtime.request({"action": "load"})
    process.terminate.assert_called_once()
    assert process.stdin.closed and process.stdout.closed


def test_unresponsive_runtime_is_killed_on_close(monkeypatch, tmp_path):
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("runtime", 5), None]
    monkeypatch.setattr(isolated.subprocess, "Popen", lambda *args, **kwargs: process)
    runtime = isolated.RuntimeProcess(["python", "handler.py"], cwd=tmp_path, env={})
    runtime.close()
    process.terminate.assert_called_once()
    process.kill.assert_called_once()


def test_request_timeout_reaps_owned_process(monkeypatch, tmp_path):
    process = Mock()
    process.poll.return_value = None
    inbox = Mock()
    inbox.get.side_effect = isolated.queue.Empty()
    monkeypatch.setattr(isolated.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(isolated.queue, "Queue", lambda **kwargs: inbox)
    monkeypatch.setattr(isolated.threading, "Thread", Mock())
    runtime = isolated.RuntimeProcess(["python", "handler.py"], cwd=tmp_path, env={})
    with pytest.raises(RuntimeError, match="timed out"):
        runtime.request({"action": "load"}, timeout=1)
    process.terminate.assert_called_once()
    process.wait.assert_called_once()


@pytest.mark.parametrize("values", [[], [float("nan")], [0.0]])
def test_runtime_rejects_empty_invalid_or_silent_speech(values, tmp_path):
    wave = Mock()
    wave.detach.return_value = wave.float.return_value = wave.cpu.return_value = wave
    wave.numpy.return_value = np.asarray(values, dtype=np.float32)
    session = CosyVoiceSession()
    session.output_root = tmp_path
    session.model = SimpleNamespace(
        sample_rate=24000,
        inference_zero_shot=Mock(return_value=iter([{"tts_speech": wave}])),
    )
    with pytest.raises(ValueError, match="empty, invalid or silent"):
        session.synthesize({"text": "Approved words.", "output": str(tmp_path / "speech.wav")})
    assert not (tmp_path / "speech.wav").exists()


def test_runtime_rejects_output_outside_private_directory(tmp_path):
    session = CosyVoiceSession()
    session.output_root = tmp_path
    session.model = Mock()
    with pytest.raises(ValueError, match="private audio directory"):
        session.synthesize(
            {"text": "Approved words.", "output": str(tmp_path.parent / "outside.wav")}
        )
    session.model.inference_zero_shot.assert_not_called()


def test_generic_runtime_server_keeps_sdk_output_out_of_json_transport(monkeypatch):
    incoming = io.StringIO('{"action": "load"}\n{"action": "close"}\n')
    outgoing = io.StringIO()
    diagnostics = io.StringIO()
    monkeypatch.setattr(isolated_runtime.sys, "stdin", incoming)
    monkeypatch.setattr(isolated_runtime.sys, "stdout", outgoing)
    monkeypatch.setattr(isolated_runtime.sys, "stderr", diagnostics)
    monkeypatch.setattr(isolated_runtime.logging, "disable", Mock())

    def load(request):
        print("SDK diagnostic")
        return {"device": "fake-device"}

    handler = SimpleNamespace(load=load)
    isolated_runtime.serve(handler)
    responses = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert responses[0] == {"ok": True, "result": {"device": "fake-device"}}
    assert responses[1]["ok"] is False
    assert diagnostics.getvalue() == "SDK diagnostic\n"


def test_runtime_session_caches_reference_and_preserves_text_and_native_rate(
    setup, monkeypatch, tmp_path
):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "existing")
    voice, audio, model_dir = setup
    component = SimpleNamespace(
        parameters=lambda: iter(
            [SimpleNamespace(device=SimpleNamespace(type="cpu"), dtype="torch.float32")]
        )
    )
    wave = Mock()
    wave.detach.return_value = wave.float.return_value = wave.cpu.return_value = wave
    wave.numpy.return_value = np.full((1, 8), 0.1, dtype=np.float32)
    model = SimpleNamespace(
        model=SimpleNamespace(llm=component, flow=component, hift=component),
        sample_rate=16000,
        add_zero_shot_spk=Mock(return_value=True),
        inference_zero_shot=Mock(return_value=iter([{"tts_speech": wave}, {"tts_speech": wave}])),
    )
    loader = Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), __version__="fake"),
    )
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", SimpleNamespace(AutoModel=loader))
    session = CosyVoiceSession()
    info = session.load(
        {
            "source": str(tmp_path),
            "model": str(model_dir),
            "reference": str(audio / voice.reference_audio),
            "reference_text": voice.reference_text,
            "device": "cpu",
            "output_root": str(tmp_path),
        }
    )
    output = tmp_path / "speech.wav"
    session.synthesize({"text": "Approved words.", "output": str(output)})
    wave, sample_rate = sf.read(output)
    assert len(wave) == 16 and sample_rate == 16000 and info["device"] == "cpu"
    model.add_zero_shot_spk.assert_called_once_with(
        voice.reference_text, str(audio / voice.reference_audio), "yapit"
    )
    model.inference_zero_shot.assert_called_once_with(
        "Approved words.",
        "",
        "",
        zero_shot_spk_id="yapit",
        stream=False,
        speed=1.0,
        text_frontend=False,
    )


def test_explicit_cuda_does_not_silently_fall_back_to_cpu(setup, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "existing")
    loader = Mock()
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", SimpleNamespace(AutoModel=loader))
    with pytest.raises(RuntimeError, match="CUDA mode is unavailable"):
        CosyVoiceSession().load({"source": str(tmp_path), "device": "cuda:0"})
    loader.assert_not_called()
