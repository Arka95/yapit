from __future__ import annotations

import gc
import getpass
import hashlib
import importlib.metadata
import json
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import numpy as np
import soundfile as sf

from yapit.common_utils.mediautils import SAMPLE_RATE
from yapit.enrollment.styles import select_voice
from yapit.models import Voice
from yapit.storage import DataStore
from yapit.yapper.speech import QwenSpeech

MAX_REQUEST = 1024 * 1024
MAX_AUDIO = 45 * SAMPLE_RATE * 4
CACHE_BYTES = 250 * 1024 * 1024


def state_path(data: DataStore) -> Path:
    return data.root / ".runtime" / "voice-worker.json"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Local voice worker must not redirect requests.")


def read_state(data: DataStore) -> dict | None:
    path = state_path(data)
    if not path.exists():
        return None
    if not path.resolve().is_relative_to(data.root) or path.stat().st_size > 4096:
        raise ValueError("Invalid local voice-worker state.")
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state.get("port"), int) or not 0 < state["port"] < 65536:
        raise ValueError("Invalid voice-worker port.")
    if not isinstance(state.get("token"), str) or len(state["token"]) < 32:
        raise ValueError("Invalid voice-worker authentication.")
    return state


def _request(state: dict, route: str, payload: dict | None = None) -> tuple[bytes, dict]:
    body = json.dumps(payload).encode() if payload is not None else None
    if body is not None and len(body) > MAX_REQUEST:
        raise ValueError("Voice-worker request is too large.")
    request = Request(
        f"http://127.0.0.1:{state['port']}{route}",
        data=body,
        headers={
            "Authorization": f"Bearer {state['token']}",
            "Content-Type": "application/json",
        },
    )
    opener = build_opener(ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=240) as response:
            content = response.read(MAX_AUDIO + 1)
            if len(content) > MAX_AUDIO:
                raise RuntimeError("Oversized voice-worker response.")
            return content, dict(response.headers)
    except HTTPError as exc:
        message = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"Voice worker rejected the request ({exc.code}): {message}") from exc
    except URLError as exc:
        raise RuntimeError(
            "Voice worker is unavailable; use voice-worker-status/stop. No second model was loaded."
        ) from exc


class WorkerSpeech:
    def __init__(self, voice: Voice, voice_dir: Path, state: dict):
        self.voice = voice
        self.voice_dir = voice_dir
        self.state = state

    def synthesize(self, text: str) -> np.ndarray:
        content, _ = _request(
            self.state,
            "/speech",
            {
                "voice": self.voice.model_dump(mode="json"),
                "voice_dir": str(self.voice_dir.resolve()),
                "text": text,
                "cache": os.environ.get("YAPIT_SPEECH_CACHE", "1") != "0",
            },
        )
        if not content or len(content) % 4:
            raise RuntimeError("Invalid voice-worker audio.")
        wave = np.frombuffer(content, dtype="<f4").copy()
        if not np.isfinite(wave).all():
            raise RuntimeError("Invalid voice-worker waveform.")
        return wave


def backend_for(voice: Voice, voice_dir: Path, *, fallback=QwenSpeech):
    if os.environ.get("YAPIT_VOICE_WORKER", "auto") != "0":
        state = read_state(DataStore())
        if state is not None:
            return WorkerSpeech(voice, voice_dir, state)
    return fallback(voice, voice_dir)


class SpeechEngine:
    def __init__(self, data: DataStore, factory=None):
        self.data = data
        self.factory = factory or QwenSpeech
        self.backend = None
        self.identity = None
        self.lock = threading.Lock()
        self.status = {"ready": False, "busy": False}
        self.cache = data.root / ".runtime" / "speech-cache"
        if not self.cache.resolve().is_relative_to(data.root):
            raise ValueError("Speech cache must stay inside the data root.")
        self.cache.mkdir(parents=True, exist_ok=True)

    def _identity(self, voice: Voice, directory: Path) -> str:
        if not directory.resolve().is_relative_to(self.data.root):
            raise ValueError("Voice directory must stay inside the data root.")
        paths = (
            [
                Path(voice.checkpoint) / name
                for name in ("model.safetensors", "config.json", "generation_config.json")
            ]
            if voice.checkpoint
            else [directory / voice.reference_audio]
        )
        stamps = []
        for path in paths:
            if not path.resolve().is_relative_to(self.data.root):
                raise ValueError("Voice weights/reference must stay inside the data root.")
            if path.is_file():
                stamps.append((str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns))
        identity = (
            {"checkpoint": voice.checkpoint, "speaker": voice.speaker or voice.name}
            if voice.checkpoint
            else {
                "reference_audio": str((directory / voice.reference_audio).resolve()),
                "reference_text": voice.reference_text,
            }
        )
        value = [identity, stamps]
        try:
            version = importlib.metadata.version("qwen-tts")
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed"
        value.extend([version, os.environ.get("YAPIT_MODEL") if not voice.checkpoint else None])
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def _load(self, voice: Voice, directory: Path) -> None:
        identity = self._identity(voice, directory)
        if identity == self.identity:
            return
        self.status["ready"] = False
        self.backend = None
        self.identity = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        started = time.perf_counter()
        backend = self.factory(voice, directory)
        backend._load()
        self.backend, self.identity = backend, identity
        model = getattr(backend.model, "model", None)
        parameter = next(model.parameters()) if model is not None else None
        self.status.update(
            ready=True,
            voice=voice.name,
            checkpoint=voice.checkpoint,
            device=str(parameter.device) if parameter is not None else "test",
            dtype=str(parameter.dtype) if parameter is not None else "test",
            load_seconds=round(time.perf_counter() - started, 3),
        )

    def prewarm(self, voice: Voice, directory: Path) -> None:
        with self.lock:
            self._load(voice, directory)

    def synthesize(self, voice: Voice, directory: Path, text: str, reuse: bool) -> np.ndarray:
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Supply 1-4000 characters of speech text.")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Voice worker is loading or busy; retry after the current chunk.")
        started = time.perf_counter()
        self.status["busy"] = True
        try:
            self._load(voice, directory)
            key = hashlib.sha256(f"{self.identity}\0{text}".encode()).hexdigest()
            path = self.cache / f"{key}.wav"
            if not path.resolve().is_relative_to(self.cache.resolve()):
                raise ValueError("Cached speech must stay inside the cache directory.")
            hit = reuse and path.is_file()
            if hit:
                wave, rate = sf.read(path, dtype="float32")
                if rate != SAMPLE_RATE:
                    raise ValueError("Invalid cached speech sample rate.")
                path.touch()
            else:
                wave = np.asarray(self.backend.synthesize(text), dtype=np.float32).reshape(-1)
            if (
                not len(wave)
                or wave.ndim != 1
                or len(wave) * 4 > MAX_AUDIO
                or not np.isfinite(wave).all()
                or np.max(np.abs(wave)) < 1e-5
            ):
                raise ValueError("Empty, silent, invalid or oversized speech waveform.")
            if not hit:
                with tempfile.NamedTemporaryFile(dir=self.cache, suffix=".wav", delete=False) as f:
                    temporary = Path(f.name)
                try:
                    sf.write(temporary, wave, SAMPLE_RATE, subtype="FLOAT")
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                files = sorted(
                    (
                        p
                        for p in self.cache.glob("*.wav")
                        if len(p.stem) == 64 and all(c in "0123456789abcdef" for c in p.stem)
                    ),
                    key=lambda p: p.stat().st_mtime_ns,
                )
                size = sum(p.stat().st_size for p in files)
                for old in files:
                    if size <= CACHE_BYTES:
                        break
                    size -= old.stat().st_size
                    old.unlink()
            elapsed = time.perf_counter() - started
            self.status.update(
                last_cache_hit=hit,
                last_request_seconds=round(elapsed, 3),
                last_audio_seconds=round(len(wave) / SAMPLE_RATE, 3),
                last_real_time_factor=round(elapsed / (len(wave) / SAMPLE_RATE), 3),
            )
            return wave
        finally:
            self.status["busy"] = False
            self.lock.release()


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, engine: SpeechEngine, token: str):
        self.engine = engine
        self.token = token
        self.stopping = threading.Event()
        super().__init__(("127.0.0.1", 0), WorkerHandler)


class WorkerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _reply(self, status: int, value: dict | bytes) -> None:
        binary = isinstance(value, bytes)
        body = value if binary else json.dumps(value).encode()
        self.send_response(status)
        self.send_header(
            "Content-Type", "application/octet-stream" if binary else "application/json"
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorize(self) -> bool:
        if (
            self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}"
            or self.headers.get("Origin") is not None
        ):
            self._reply(403, {"error": "Local clients only."})
            return False
        if not secrets.compare_digest(
            self.headers.get("Authorization", "").encode("utf-8"),
            f"Bearer {self.server.token}".encode("ascii"),
        ):
            self._reply(401, {"error": "Authentication required."})
            return False
        return True

    def do_GET(self):
        if self._authorize():
            if self.path == "/health":
                self._reply(200, {"running": True, **self.server.engine.status})
            else:
                self._reply(404, {"error": "Unknown route."})

    def do_POST(self):
        if not self._authorize():
            return
        if self.path == "/stop":
            self._reply(200, {"stop_requested": True})
            self.server.stopping.set()
            return
        if self.path != "/speech":
            self._reply(404, {"error": "Unknown route."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST:
                raise ValueError("Invalid request size.")
            self.connection.settimeout(10)
            payload = json.loads(self.rfile.read(length))
            wave = self.server.engine.synthesize(
                Voice.model_validate(payload["voice"]),
                Path(payload["voice_dir"]),
                payload["text"],
                payload.get("cache", True) is True,
            )
            self._reply(200, np.asarray(wave, dtype="<f4").tobytes())
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self._reply(400, {"error": str(exc)})
        except RuntimeError as exc:
            self._reply(503, {"error": str(exc)})


def _write_state(data: DataStore, state: dict) -> None:
    path = state_path(data)
    if not path.resolve().is_relative_to(data.root):
        raise ValueError("Worker state must stay inside the data root.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        if os.name == "nt":
            subprocess.run(
                [
                    "icacls",
                    str(path),
                    "/inheritance:r",
                    "/grant:r",
                    f"{getpass.getuser()}:(F)",
                ],
                check=True,
                capture_output=True,
                timeout=10,
            )
        with os.fdopen(descriptor, "w", encoding="utf-8") as f:
            descriptor = None
            json.dump(state, f)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def worker_status(data: DataStore) -> dict:
    state = read_state(data)
    if state is None:
        return {"running": False}
    content, _ = _request(state, "/health")
    return json.loads(content)


def stop_worker(data: DataStore) -> dict:
    state = read_state(data)
    if state is None:
        return {"running": False}
    try:
        content, _ = _request(state, "/stop", {})
    except RuntimeError as exc:
        cause = exc.__cause__
        if isinstance(cause, URLError) and isinstance(cause.reason, ConnectionRefusedError):
            if read_state(data) == state:
                state_path(data).unlink()
            return {"running": False, "stale_state_removed": True}
        raise
    return json.loads(content)


def run_worker(data: DataStore, voice_name: str) -> None:
    if state_path(data).exists():
        raise RuntimeError("Worker state already exists; use voice-worker-status/stop first.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    voice, directory = select_voice(data, voice_name)
    engine = SpeechEngine(data)
    token = secrets.token_urlsafe(32)
    server = WorkerServer(engine, token)
    state = {"port": server.server_port, "token": token, "pid": os.getpid()}
    thread = None
    try:
        _write_state(data, state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print("Voice worker listening locally; loading selected weights.", flush=True)
        engine.prewarm(voice, directory)
        print(json.dumps(engine.status), flush=True)
        server.stopping.wait()
    finally:
        if thread is not None:
            server.shutdown()
        server.server_close()
        if read_state(data) == state:
            state_path(data).unlink()
