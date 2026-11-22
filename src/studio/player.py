from __future__ import annotations

import ctypes
import uuid
from pathlib import Path


class WindowsAudioPlayer:
    """Play/seek a prepared WAV through Windows MCI without an extra media SDK."""

    def __init__(self):
        self.alias = f"yapit_{uuid.uuid4().hex}"
        self.path: Path | None = None
        self._dll = None

    def _command(self, command: str) -> str:
        if self._dll is None:
            self._dll = ctypes.WinDLL("winmm")
            self._dll.mciSendStringW.argtypes = (
                ctypes.c_wchar_p,
                ctypes.c_wchar_p,
                ctypes.c_uint,
                ctypes.c_void_p,
            )
            self._dll.mciSendStringW.restype = ctypes.c_uint
        output = ctypes.create_unicode_buffer(256)
        error = self._dll.mciSendStringW(command, output, len(output), None)
        if error:
            raise RuntimeError(f"Windows audio playback failed (MCI error {error}).")
        return output.value

    def play(self, path: Path, seconds: float = 0) -> None:
        if self.path != path:
            self.close()
            self._command(f'open "{path.resolve()}" type waveaudio alias {self.alias}')
            self.path = path
            self._command(f"set {self.alias} time format milliseconds")
        self._command(f"seek {self.alias} to {max(0, round(seconds * 1000))}")
        self._command(f"play {self.alias}")

    def pause(self) -> None:
        if self.path is not None:
            self._command(f"stop {self.alias}")

    def position(self) -> tuple[float, bool]:
        if self.path is None:
            return 0, False
        seconds = int(self._command(f"status {self.alias} position")) / 1000
        return seconds, self._command(f"status {self.alias} mode") == "playing"

    def close(self) -> None:
        if self.path is not None:
            self._command(f"close {self.alias}")
            self.path = None
