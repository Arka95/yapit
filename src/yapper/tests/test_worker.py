import secrets
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
import pytest

from yapit.enrollment.styles import select_voice
from yapit.yapper.backends import asset_stamps
from yapit.yapper.worker import (
    SpeechEngine,
    WorkerServer,
    WorkerSpeech,
    _request,
    _write_state,
    backend_for,
    read_state,
    state_path,
    stop_worker,
    worker_status,
)

from .conftest import tone


class FakeSpeech:
    loads = 0
    generations = 0

    def __init__(self, voice, directory):
        self.voice = voice
        self.directory = directory

    def load(self):
        FakeSpeech.loads += 1

    def close(self):
        pass

    def cache_identity(self):
        paths = (
            [self.directory / self.voice.reference_audio]
            if self.voice.checkpoint is None
            else [Path(self.voice.checkpoint) / "model.safetensors"]
        )
        return {
            "assets": asset_stamps(paths),
            "checkpoint": self.voice.checkpoint,
            "reference_text": self.voice.reference_text,
        }

    def describe(self):
        return {"device": "test", "dtype": "test"}

    def synthesize(self, text):
        FakeSpeech.generations += 1
        return tone(1)


@pytest.fixture
def worker(enrolled):
    FakeSpeech.loads = FakeSpeech.generations = 0
    voice, directory = select_voice(enrolled.data, "test-voice")
    engine = SpeechEngine(enrolled.data, FakeSpeech)
    engine.prewarm(voice, directory)
    server = WorkerServer(engine, secrets.token_urlsafe(32))
    state = {"port": server.server_port, "token": server.token, "pid": 0}
    _write_state(enrolled.data, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield enrolled.data, voice, directory, engine, server, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        state_path(enrolled.data).unlink(missing_ok=True)


def test_auto_client_reuses_model_and_exact_waveform_cache(worker, monkeypatch):
    data, voice, directory, engine, server, state = worker
    client = backend_for(voice, directory)
    assert isinstance(client, WorkerSpeech)
    first = client.synthesize("Same words.")
    assert np.array_equal(first, client.synthesize("Same words."))
    assert FakeSpeech.loads == 1 and FakeSpeech.generations == 1
    assert worker_status(data)["last_cache_hit"]
    monkeypatch.setenv("YAPIT_SPEECH_CACHE", "0")
    client.synthesize("Same words.")
    assert FakeSpeech.loads == 1 and FakeSpeech.generations == 2
    monkeypatch.setenv("YAPIT_VOICE_WORKER", "0")
    assert isinstance(backend_for(voice, directory, fallback=FakeSpeech), FakeSpeech)


def test_checkpoint_change_reloads_and_invalidates_cached_speech(worker):
    data, voice, directory, engine, server, state = worker
    checkpoint = data.root / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"first")
    voice.checkpoint = str(checkpoint)
    engine.synthesize(voice, directory, "Same words.", True)
    weights.write_bytes(b"different weights")
    engine.synthesize(voice, directory, "Same words.", True)
    assert FakeSpeech.loads == 3 and FakeSpeech.generations == 2


def test_writing_metadata_does_not_unload_the_model(worker):
    data, voice, directory, engine, server, state = worker
    engine.synthesize(voice, directory, "Same words.", True)
    voice.writing_style = "energetic"
    voice.words_per_minute = 180
    engine.synthesize(voice, directory, "Same words.", True)
    assert FakeSpeech.loads == 1 and FakeSpeech.generations == 1


def test_worker_auth_confinement_busy_and_stop(worker):
    data, voice, directory, engine, server, state = worker
    with pytest.raises(RuntimeError, match="401"):
        _request({**state, "token": "wrong"}, "/health")
    bad_directory = directory.parent.parent.parent / "outside"
    client = WorkerSpeech(voice, bad_directory, state)
    with pytest.raises(RuntimeError, match="inside the data root"):
        client.synthesize("Hello.")
    engine.lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="busy"):
            WorkerSpeech(voice, directory, state).synthesize("Hello.")
    finally:
        engine.lock.release()
    assert stop_worker(data)["stop_requested"]
    assert server.stopping.is_set()


@pytest.mark.parametrize("headers", [{"Host": "example.org"}, {"Origin": "https://example.org"}])
def test_worker_rejects_browser_origins_and_foreign_hosts(worker, headers):
    data, voice, directory, engine, server, state = worker
    request = Request(
        f"http://127.0.0.1:{state['port']}/health",
        headers={"Authorization": f"Bearer {state['token']}", **headers},
    )
    with pytest.raises(HTTPError) as error:
        build_opener(ProxyHandler({})).open(request, timeout=2)
    assert error.value.code == 403


def test_unavailable_worker_does_not_load_a_second_model(worker, monkeypatch):
    data, voice, directory, engine, server, state = worker
    server.shutdown()
    server.server_close()
    before = FakeSpeech.loads
    with pytest.raises(RuntimeError, match="No second model"):
        backend_for(voice, directory, fallback=FakeSpeech).synthesize("Hello.")
    assert FakeSpeech.loads == before
    assert stop_worker(data)["stale_state_removed"]
    assert read_state(data) is None


def test_absent_worker_preserves_one_shot_behavior(enrolled):
    voice, directory = select_voice(enrolled.data, "test-voice")
    assert isinstance(backend_for(voice, directory, fallback=FakeSpeech), FakeSpeech)
