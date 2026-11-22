from __future__ import annotations

import importlib.metadata
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yapit.common_utils.mediautils import (
    SAMPLE_RATE,
    decode_audio,
    validate_media_source,
    write_wav,
)
from yapit.models import Timeline
from yapit.storage import Workspace
from yapit.yapper.backends import BUILTINS, FACTORIES


def backend_names() -> list[str]:
    """Discover trusted adapter names without importing their SDKs."""
    plugins = importlib.metadata.entry_points(group="yapit.tts")
    return sorted(set(BUILTINS) | set(FACTORIES) | {entry.name for entry in plugins})


def waveform_peaks(wave: np.ndarray, bins: int = 800) -> list[float]:
    if not len(wave):
        raise ValueError("Audio is empty.")
    return [float(np.max(np.abs(part))) for part in np.array_split(wave, min(bins, len(wave)))]


@dataclass(frozen=True)
class AudioPreview:
    audio: Path
    playback: Path
    seconds: float
    peaks: list[float]
    timeline: Timeline | None


class StudioSession:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.preview: AudioPreview | None = None
        self.video: Path | None = None
        self._scratch = tempfile.TemporaryDirectory(prefix="studio-", dir=workspace.root)
        self._revision = 0

    def load_audio(self, audio: Path, timeline_path: Path | None = None) -> AudioPreview:
        audio = validate_media_source(audio)
        timeline = (
            Timeline.model_validate_json(timeline_path.read_text(encoding="utf-8"))
            if timeline_path is not None
            else None
        )
        wave = decode_audio(audio, max_seconds=timeline.seconds + 1 if timeline else 180)
        seconds = len(wave) / SAMPLE_RATE
        if timeline and abs(timeline.seconds - seconds) > 0.1:
            raise ValueError("Timeline duration does not match the loaded audio.")
        self._revision += 1
        playback = Path(self._scratch.name) / f"preview-{self._revision}.wav"
        write_wav(playback, wave)
        preview = AudioPreview(audio, playback, seconds, waveform_peaks(wave), timeline)
        old = self.preview
        self.preview, self.video = preview, None
        if old is not None:
            old.playback.unlink(missing_ok=True)
        return preview

    def analyze_beats(self) -> AudioPreview:
        from yapit.rescript.transcription import Segment, timeline_from_segments, transcribe

        if self.preview is None:
            raise ValueError("Load audio first.")
        preview = self.preview
        if preview.timeline is not None:
            return preview
        result = transcribe(
            preview.audio,
            data=self.workspace.data,
            max_seconds=180,
            offline=True,
            export_scripts=False,
        )
        metadata = json.loads(Path(result["transcript"]).read_text(encoding="utf-8"))
        timeline = timeline_from_segments(
            [Segment.model_validate(segment) for segment in metadata["segments"]],
            result["source_seconds"],
        )
        self.preview = AudioPreview(
            preview.audio, preview.playback, preview.seconds, preview.peaks, timeline
        )
        return self.preview

    def generate_video(self) -> dict:
        from yapit.editor import generate_video

        if self.preview is None or self.preview.timeline is None:
            raise ValueError("Analyze narration beats before generating the video template.")
        preview = self.preview
        result = generate_video(
            preview.timeline,
            preview.audio,
            workspace=self.workspace,
            actual_seconds=preview.seconds,
        )
        self.video = Path(result["video"])
        return result

    def close(self) -> None:
        self._scratch.cleanup()
