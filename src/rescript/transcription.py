from __future__ import annotations

import os
import re
import sys
import tempfile
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from typing import Annotated, Protocol

import soundfile as sf
from pydantic import Field, ValidationError, model_validator

from yapit.common_utils.fileutils import reserve_artifact_name, save_json
from yapit.common_utils.mediautils import extract_audio_stream, has_video, validate_media_source
from yapit.common_utils.metrics import timed
from yapit.common_utils.modelutils import load_model
from yapit.constants import DEFAULT_ASR_MODEL, DEFAULT_PROFILE
from yapit.models import Beat, Cue, Model, Script, Timeline
from yapit.rescript.audio import extract_audio
from yapit.storage import DataStore

MAX_INPUT_SECONDS = 3600
SENTENCE_END = re.compile(r"""[.!?]["')\]]*$""")


class Segment(Model):
    start: Annotated[float, Field(ge=0)]
    end: Annotated[float, Field(gt=0)]
    text: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def positive_duration(self) -> Segment:
        if self.end <= self.start:
            raise ValueError("Transcript segments must have positive duration.")
        return self


class TranscriptionBackend(Protocol):
    def recognize(self, audio: Path) -> list[Segment]: ...


class WhisperTranscriber:
    def __init__(
        self,
        model: str = DEFAULT_ASR_MODEL,
        device: str | None = None,
        compute_type: str | None = None,
        language: str = "en",
        offline: bool = True,
    ):
        self.model = model
        self.device = device or os.getenv("YAPIT_ASR_DEVICE", "cpu")
        self.compute_type = (
            compute_type
            or (os.getenv("YAPIT_ASR_COMPUTE_TYPE") if device is None else None)
            or ("float16" if self.device == "cuda" else "int8")
        )
        self.language = language
        self.offline = offline

    @timed
    def recognize(self, audio: Path) -> list[Segment]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                'Install transcription support: python -m pip install -e ".[rescript]"'
            ) from exc
        with redirect_stdout(sys.stderr):
            model = load_model(
                WhisperModel,
                self.model,
                cache_parameter="download_root",
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=self.offline,
            )
            segments, _ = model.transcribe(
                str(audio),
                language=self.language,
                task="transcribe",
                vad_filter=True,
                word_timestamps=True,
                condition_on_previous_text=False,
            )
            result = []
            words = []
            for segment in segments:
                if not segment.text.strip():
                    continue
                if not segment.words:
                    raise ValueError("Recognition returned speech without word timestamps.")
                for word in segment.words:
                    if not word.word.strip():
                        continue
                    words.append(word)
                    if SENTENCE_END.search(word.word.strip()) or len(words) >= 35:
                        result.append(self._segment(words))
                        words = []
            if words:
                result.append(self._segment(words))
        return result

    @staticmethod
    def _segment(words: list) -> Segment:
        return Segment(
            start=words[0].start,
            end=words[-1].end,
            text=" ".join(word.word.strip() for word in words),
        )


@timed
def compose_scripts(segments: list[Segment], title: str) -> list[tuple[Script, float, float]]:
    """Partition at sentence endings without discarding or repeating recognized text."""
    if not segments:
        raise ValueError("No speech was detected; no script was written.")
    previous = 0.0
    for segment in segments:
        if segment.start < previous:
            raise ValueError("Transcript timestamps must be ordered and non-overlapping.")
        previous = segment.end
    origin = segments[0].start
    beats = [
        Beat(
            text=segment.text,
            visual="Supporting B-roll",
            pause_ms=(
                min(1200, round((segments[index + 1].start - segment.end) * 1000))
                if index + 1 < len(segments)
                else 0
            ),
        )
        for index, segment in enumerate(segments)
    ]
    boundaries = [(0, origin)]
    boundaries.extend(
        (index, segment.end)
        for index, segment in enumerate(segments[:-1], start=1)
        if SENTENCE_END.search(segment.text)
    )
    boundaries.append((len(segments), segments[-1].end))
    # Keep globally viable cuts: a greedy 60s cut can leave an unusable <=30s tail.
    best: dict[int, tuple[int, float, int, Script]] = {}
    for right in range(1, len(boundaries)):
        end_index, end = boundaries[right]
        for left in range(right - 1, -1, -1):
            start_index, start = boundaries[left]
            seconds = end - start
            if seconds > 60:
                break
            if (left != 0 or right != len(boundaries) - 1) and seconds <= 30:
                continue
            if left and left not in best:
                continue
            try:
                selected = [beat.model_copy() for beat in beats[start_index:end_index]]
                selected[-1].pause_ms = 0
                script = Script(title=title[:200], beats=selected)
            except ValidationError:
                continue
            count = best[left][0] + 1 if left else 1
            cost = (best[left][1] if left else 0) + (seconds - 45) ** 2
            candidate = (count, cost, left, script)
            if right not in best or candidate[:2] < best[right][:2]:
                best[right] = candidate
    right = len(boundaries) - 1
    if right not in best:
        raise ValueError(
            "Cannot form YapIt scripts at sentence boundaries: each split part must be "
            "over 30 and at most 60 seconds, with at most 40 beats and 650 words. "
            "Review the recording or supply shorter, meaningfully separated clips."
        )
    result = []
    while right:
        _, _, left, script = best[right]
        result.append((script, boundaries[left][1], boundaries[right][1]))
        right = left
    return list(reversed(result))


@timed
def transcribe(
    media: Path,
    *,
    output: Path | None = None,
    model: str = DEFAULT_ASR_MODEL,
    device: str | None = None,
    compute_type: str | None = None,
    language: str = "en",
    offline: bool = True,
    backend: TranscriptionBackend | None = None,
    profile: str = DEFAULT_PROFILE,
    data: DataStore | None = None,
    max_seconds: float = MAX_INPUT_SECONDS,
) -> dict:
    if not 0 < max_seconds <= MAX_INPUT_SECONDS:
        raise ValueError(f"max_seconds must be positive and at most {MAX_INPUT_SECONDS}.")
    source = validate_media_source(media)
    data = data or DataStore()
    destination = data.within(data.transcripts, output or data.transcripts)
    if destination != data.transcripts:
        raise ValueError("Transcript output must be data\\transcripts; subfolders are not created.")
    recognizer = backend or WhisperTranscriber(model, device, compute_type, language, offline)
    with tempfile.TemporaryDirectory(prefix="yapit-rescript-") as temporary:
        audio = Path(temporary) / "audio.wav"
        decoded = extract_audio_stream(source, audio, sample_rate=16000, max_seconds=max_seconds)
        info = sf.info(audio)
        if not 0 < info.duration <= max_seconds:
            raise ValueError(
                f"Audio must be nonempty and at most {max_seconds:g} seconds; "
                "no truncation allowed."
            )
        if has_video(decoded.stderr):
            archived = Path(
                extract_audio(
                    source, output_dir=data.audio / "imports" / uuid.uuid4().hex, profile=profile
                )["audio"]
            )
        else:
            archived = data.archive_recording(source, profile)
        segments = recognizer.recognize(audio)
        if any(segment.end > info.duration + 0.1 for segment in segments):
            raise ValueError("Transcript timestamps extend past the decoded audio.")
        parts = compose_scripts(segments, source.stem or "Transcript")
    name = reserve_artifact_name(
        source.stem,
        profile,
        [data.transcripts, data.generated_audio],
        destination,
        "transcript.json",
    )
    outputs = []
    for index, (script, start, end) in enumerate(parts, start=1):
        suffix = "script.json" if len(parts) == 1 else f"script-{index:03}.json"
        if len(parts) > 1:
            script.title = f"{script.title[:180]} - Part {index}"
        path = destination / f"{name}.{suffix}"
        save_json(path, script)
        outputs.append(
            {
                "script": str(path),
                "source_start": start,
                "source_end": end,
                "source_seconds": end - start,
                "suggested_seconds": max(5, end - start),
            }
        )
    result = {
        "status": "needs_review",
        "directory": str(destination),
        "name": name,
        "transcript": str(destination / f"{name}.transcript.json"),
        "source": str(source),
        "source_seconds": info.duration,
        "audio": str(archived),
        "model": model if backend is None else "injected",
        "language": language,
        "timing_method": "ASR word timestamps; source timing, not synthesized duration",
        "review_required": True,
        "scripts": outputs,
    }
    save_json(
        destination / f"{name}.transcript.json",
        {**result, "segments": [segment.model_dump(mode="json") for segment in segments]},
    )
    return result


def extract_script(
    media_path: Path,
    *,
    model: str = DEFAULT_ASR_MODEL,
    language: str = "en",
    offline: bool = True,
    data: DataStore | None = None,
) -> dict:
    """Extract reviewable Script JSON and source timings from local audio/video."""
    result = transcribe(media_path, model=model, language=language, offline=offline, data=data)
    return {
        **result,
        "scripts": [
            {
                **part,
                "content": Script.model_validate_json(
                    Path(part["script"]).read_text(encoding="utf-8")
                ).model_dump(mode="json"),
            }
            for part in result["scripts"]
        ],
    }


def timeline_from_segments(segments: list[Segment], seconds: float) -> Timeline:
    """Retain ASR starts while including leading, inter-phrase and trailing silence."""
    if not segments:
        raise ValueError("No speech was detected; cannot build a reel timeline.")
    previous = 0.0
    for segment in segments:
        if segment.start < previous or segment.start >= seconds or segment.end > seconds + 0.1:
            raise ValueError("Transcript timestamps must be ordered and fit the source audio.")
        previous = segment.end
    return Timeline(
        seconds=seconds,
        timing_method="ASR segment timestamps; original audio preserved",
        cues=[
            Cue(
                start=segment.start if index else 0,
                end=segments[index + 1].start if index + 1 < len(segments) else seconds,
                text=segment.text,
                visual="Supporting B-roll",
                beat=index,
            )
            for index, segment in enumerate(segments)
        ],
    )
