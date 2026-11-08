from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path

MAX_MESSAGE = 65536


class RuntimeProcess:
    """Bounded local JSON transport for adapters with incompatible interpreters."""

    def __init__(self, command: list[str], *, cwd: Path, env: dict[str, str]):
        self.lock = threading.RLock()
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env={**env, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    def request(self, payload: dict, *, timeout: float = 180) -> dict:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("Isolated model runtime is not running.")
            received = queue.Queue(maxsize=1)
            stream = self.process.stdout

            def read() -> None:
                try:
                    received.put(stream.readline(MAX_MESSAGE + 1))
                except (OSError, ValueError) as exc:
                    received.put(exc)

            try:
                message = json.dumps(payload, ensure_ascii=True) + "\n"
                if len(message) > MAX_MESSAGE:
                    raise ValueError("Isolated model request is too large.")
                self.process.stdin.write(message)
                self.process.stdin.flush()
                threading.Thread(target=read, daemon=True).start()
                line = received.get(timeout=timeout)
                if isinstance(line, Exception):
                    raise line
                if not line or len(line) > MAX_MESSAGE or not line.endswith("\n"):
                    raise RuntimeError("Isolated model returned an invalid response.")
                response = json.loads(line)
                if not isinstance(response, dict) or response.get("ok") is not True:
                    error = response.get("error") if isinstance(response, dict) else None
                    raise RuntimeError(error or "Isolated model request failed.")
                if not isinstance(response.get("result"), dict):
                    raise RuntimeError("Isolated model returned invalid metadata.")
                return response["result"]
            except queue.Empty as exc:
                self.close()
                raise RuntimeError("Isolated model request timed out.") from exc
            except (OSError, ValueError, RuntimeError):
                self.close()
                raise

    def close(self) -> None:
        with self.lock:
            process, self.process = self.process, None
            if process is None:
                return
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            finally:
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        stream.close()
