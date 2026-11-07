from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from yapit.media import SAMPLE_RATE, ffmpeg


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

    @property
    def captured_seconds(self) -> float:
        if self.output is None:
            return 0.0
        progress = self.output.with_suffix(".capture.log").read_text(
            encoding="utf-8", errors="replace"
        )
        timestamps = re.findall(r"^out_time_us=(\d+)$", progress, flags=re.MULTILINE)
        return int(timestamps[-1]) / 1_000_000 if timestamps else 0.0

    def start(self, device: str, output: Path) -> None:
        if os.name != "nt":
            raise RuntimeError("Recording is supported on Windows only.")
        if self.process is not None and self.process.poll() is None:
            raise ValueError("A recording is already running.")
        if not device:
            raise ValueError("Select a microphone first.")
        if output.exists():
            raise ValueError("Refusing to overwrite an existing raw take.")
        self.output = output
        self.log = output.with_suffix(".capture.log").open("wb")
        try:
            self.process = subprocess.Popen(
                [
                    ffmpeg(),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostats",
                    "-stats_period",
                    "0.1",
                    "-progress",
                    "pipe:2",
                    "-n",
                    "-f",
                    "dshow",
                    "-audio_buffer_size",
                    "50",
                    "-i",
                    f"audio={device}",
                    "-t",
                    "20",
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-c:a",
                    "pcm_s16le",
                    "-flush_packets",
                    "1",
                    str(output),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.log,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except Exception:
            self.log.close()
            raise

    def stop(self) -> Path:
        if self.process is None or self.output is None:
            raise ValueError("There is no active recording.")
        process = self.process
        try:
            process.communicate(input=b"q\n" if process.poll() is None else None, timeout=5)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait(timeout=3)
            raise RuntimeError(
                "The microphone did not stop cleanly. Please record another take."
            ) from exc
        finally:
            if self.log:
                self.log.close()
        if process.returncode:
            detail = self.output.with_suffix(".capture.log").read_text(
                encoding="utf-8", errors="replace"
            )[-2000:]
            raise RuntimeError("Microphone recording failed. " + detail)
        return self.output
