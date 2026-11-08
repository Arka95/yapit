from __future__ import annotations

import os
import re
import subprocess
import threading
import time
import wave
from pathlib import Path

from yapit.common_utils.fileutils import data_filename
from yapit.common_utils.mediautils import SAMPLE_RATE, ffmpeg
from yapit.common_utils.metrics import timed


def microphone_devices() -> list[str]:
    if os.name != "nt":
        raise RuntimeError("The desktop recorder currently supports Windows microphones only.")
    result = subprocess.run(
        [ffmpeg(), "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True,
        timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    text = result.stderr.decode("utf-8", errors="replace")
    return list(dict.fromkeys(re.findall(r'"([^"]+)"\s+\(audio\)', text)))


class MicrophoneRecorder:
    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.output: Path | None = None
        self.log = None
        self.device = ""
        self.log_path: Path | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._preroll = bytearray()
        self._writer = None
        self._file = None
        self._frames = 0
        self._closing = False
        self._error = ""

    @property
    def captured_seconds(self) -> float:
        with self._lock:
            return self._frames / SAMPLE_RATE

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self.process is not None and self.process.poll() is None

    @timed
    def prepare(self, device: str, directory: Path) -> None:
        if os.name != "nt":
            raise RuntimeError("Recording is supported on Windows only.")
        if not device:
            raise ValueError("Select a microphone first.")
        directory = directory.resolve()
        if self.ready and device == self.device and self.log_path.parent == directory:
            return
        with self._lock:
            if self._writer is not None:
                raise ValueError("Stop the recording before changing the microphone.")
        self.close()
        directory.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.log_path = directory / data_filename("microphone", directory.name, ".capture.log")
        self.log = self.log_path.open("wb")
        self._closing = False
        self._error = ""
        self._preroll.clear()
        try:
            self.process = subprocess.Popen(
                [
                    ffmpeg(),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostats",
                    "-f",
                    "dshow",
                    "-audio_buffer_size",
                    "50",
                    "-i",
                    f"audio={device}",
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-c:a",
                    "pcm_s16le",
                    "-flush_packets",
                    "1",
                    "-f",
                    "s16le",
                    "pipe:1",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                bufsize=0,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            self._thread = threading.Thread(target=self._capture, daemon=True)
            self._thread.start()
            deadline = time.monotonic() + 15
            while not self._ready.wait(0.05):
                if not self._thread.is_alive() or time.monotonic() >= deadline:
                    detail = (
                        self._error
                        or self.log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                    )
                    raise RuntimeError("Microphone did not become ready. " + detail)
        except Exception:
            self.close()
            raise

    def _capture(self) -> None:
        try:
            pending = b""
            while chunk := self.process.stdout.read(SAMPLE_RATE // 10):
                pending += chunk
                size = len(pending) // 2 * 2
                if not size:
                    continue
                data, pending = pending[:size], pending[size:]
                with self._lock:
                    self._preroll.extend(data)
                    # Keep 250 ms in memory to preserve speech beginning at the click.
                    del self._preroll[: max(0, len(self._preroll) - SAMPLE_RATE // 2)]
                    if self._writer is not None:
                        self._write(data)
                self._ready.set()
            if not self._closing:
                self._error = "The microphone stream stopped; prepare it again."
        except Exception as exc:
            self._error = str(exc)
        finally:
            self._ready.clear()
            with self._lock:
                self._stop_writer()

    def _write(self, data: bytes) -> None:
        remaining = 20 * SAMPLE_RATE - self._frames
        data = data[: remaining * 2]
        self._writer.writeframes(data)
        self._file.flush()
        self._frames += len(data) // 2
        if self._frames == 20 * SAMPLE_RATE:
            self._stop_writer()

    def _stop_writer(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None
                self._file.close()
                self._file = None

    @timed
    def start(self, device: str, output: Path) -> None:
        if output.exists():
            raise ValueError("Refusing to overwrite an existing raw take.")
        with self._lock:
            if self._writer is not None:
                raise ValueError("A recording is already running.")
        if not self.ready or device != self.device:
            self.prepare(device, output.parent)
        with self._lock:
            if self._writer is not None:
                raise ValueError("A recording is already running.")
            self._file = output.open("xb")
            try:
                self._writer = wave.open(self._file, "wb")
                self._writer.setnchannels(1)
                self._writer.setsampwidth(2)
                self._writer.setframerate(SAMPLE_RATE)
                self.output = output
                self._frames = 0
                self._write(bytes(self._preroll))
            except Exception:
                if self._writer is not None:
                    self._stop_writer()
                else:
                    self._file.close()
                    self._file = None
                raise

    @timed
    def stop(self) -> Path:
        if self.output is None:
            raise ValueError("There is no active recording.")
        with self._lock:
            self._stop_writer()
        if self._error:
            raise RuntimeError("Microphone recording failed. " + self._error)
        return self.output

    def close(self) -> None:
        self._closing = True
        self._ready.clear()
        with self._lock:
            self._stop_writer()
        process = self.process
        if process is None:
            if self.log:
                self.log.close()
            return
        try:
            if process.poll() is None:
                try:
                    process.stdin.write(b"q\n")
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        finally:
            if self._thread:
                self._thread.join(timeout=3)
            process.stdin.close()
            process.stdout.close()
            if self.log:
                self.log.close()
            self.process = None
