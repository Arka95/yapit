from __future__ import annotations

import hashlib
import os
import re
import subprocess
import uuid
from pathlib import Path

import numpy as np
from pydantic import Field, model_validator

from yapit.media import SAMPLE_RATE, decode_audio, ffmpeg, write_wav
from yapit.models import Model, Sample, Voice
from yapit.storage import Workspace, safe_name, save_json
from yapit.voices import audio_quality, enroll


class Reading(Model):
    title: str
    text: str


READINGS = [
    Reading(
        title="Natural conversation",
        text="Here's what I wish I'd known before booking this trip. You don't need a packed "
        "schedule; you need a few good choices and some breathing room.",
    ),
    Reading(
        title="Place names and travel",
        text="We started in Bengaluru, flew to Kochi, and drove towards Munnar. The winding "
        "roads were beautiful, but I'd leave early and keep the first evening free.",
    ),
    Reading(
        title="Directions and connected speech",
        text="Take the quieter route through the village, cross the narrow bridge, and turn "
        "right after the bakery. There's a small viewpoint just beyond the trees.",
    ),
    Reading(
        title="Times and numbers",
        text="The train leaves at seven forty-five. We paid two thousand three hundred rupees "
        "for the room, plus eight hundred and fifty for breakfast and parking.",
    ),
    Reading(
        title="A curious question",
        text="Would I wake up before sunrise for this view again? Absolutely. Watching the "
        "clouds drift beneath the ridge was easily my favourite part of the weekend.",
    ),
    Reading(
        title="A balanced product review",
        text="This microphone sounds clear, feels sturdy, and fits comfortably in my backpack. "
        "The battery lasts well, though the charging cable is frustratingly short.",
    ),
    Reading(
        title="Contrast and emphasis",
        text="I tried both cameras in bright sunlight and a dimly lit kitchen. The cheaper "
        "one handled colours nicely, but struggled with shadows and quick movement.",
    ),
    Reading(
        title="An honest verdict",
        text="Honestly, I wouldn't buy it just for that feature. If you'll use it every week, "
        "it makes sense. Otherwise, keep what you already have.",
    ),
    Reading(
        title="Everyday detail",
        text="Three friends shared fresh bread, warm soup, and a surprisingly strong cup of "
        "coffee. We talked for a while, then walked slowly back through the square.",
    ),
    Reading(
        title="Measurements and prices",
        text="It weighs two hundred and fifty grams, charges in ninety minutes, and costs "
        "four thousand nine hundred rupees. That's useful, but check if it suits your routine.",
    ),
    Reading(
        title="A calm explanation",
        text="You can change the plan if the weather turns. Choose a nearby museum, save the "
        "longer walk for tomorrow, and give yourself enough time to get back.",
    ),
    Reading(
        title="A warm closing",
        text="If you're choosing between doing more and enjoying it more, I'd slow down. "
        "Leave space for an unexpected conversation, a good meal, or simply a quiet evening.",
    ),
]


class Draft(Model):
    alias: str
    consent: bool
    readings: list[Reading] = Field(min_length=1, max_length=100)
    samples: list[Sample | None]
    completed: bool = False

    @model_validator(mode="after")
    def matching_samples(self) -> Draft:
        safe_name(self.alias)
        if len(self.readings) != len(self.samples):
            raise ValueError("Enrollment draft has inconsistent sample slots.")
        return self


class EnrollmentSession:
    def __init__(self, workspace: Workspace, alias: str, consent: bool):
        if not consent:
            raise ValueError("Confirm that this is your own voice and you consent to cloning.")
        self.workspace = workspace
        self.alias = safe_name(alias.strip())
        if workspace.voice_path(self.alias).exists():
            raise ValueError(
                f"Alias '{self.alias}' is already enrolled. Choose a new alias, such as "
                f"'{self.alias}-v2'; existing recordings will not be overwritten."
            )
        self.folder = workspace.inside(Path("enrollment") / self.alias, exists=False)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "session.json"
        if self.path.exists():
            self.draft = Draft.model_validate_json(self.path.read_text(encoding="utf-8"))
            if self.draft.alias != self.alias:
                raise ValueError("The saved enrollment alias does not match this folder.")
        else:
            self.draft = Draft(
                alias=self.alias,
                consent=True,
                readings=[reading.model_copy() for reading in READINGS],
                samples=[None] * len(READINGS),
            )
        self.draft.consent = True
        save_json(self.path, self.draft)

    @property
    def index(self) -> int:
        return next(
            (index for index, sample in enumerate(self.draft.samples) if sample is None),
            len(self.draft.samples),
        )

    @property
    def finished(self) -> bool:
        return self.index == len(self.draft.readings)

    def new_take(self) -> Path:
        if self.finished:
            raise ValueError("All passages are saved; finish enrollment or choose another alias.")
        directory = self.folder / "takes"
        directory.mkdir(exist_ok=True)
        return directory / f"raw-{self.index + 1:02}-{uuid.uuid4().hex[:12]}.wav"

    def save_take(self, source: Path, confirmed: bool, normalize: bool) -> dict:
        if not confirmed:
            raise ValueError(
                "Replay the take and confirm that you read the displayed text exactly."
            )
        if self.finished:
            raise ValueError("All passages are already saved.")
        source = self.workspace.inside(source)
        if not source.is_relative_to(self.folder):
            raise ValueError("The recording does not belong to this enrollment session.")
        data = decode_audio(source, max_seconds=20)
        original_quality = audio_quality(data)
        if original_quality["seconds"] < 3:
            raise ValueError("Record at least 3 seconds and read the complete passage.")
        if original_quality["clipped_fraction"] > 0.001:
            raise ValueError("This take is clipped. Reduce microphone gain and record again.")
        gain_db = 0.0
        trim_start = trim_end = 0.0
        if normalize:
            window = SAMPLE_RATE // 50
            frames = data[: len(data) // window * window].reshape(-1, window)
            active = np.sqrt(np.mean(frames**2, axis=1)) > 0.002
            sustained = (
                np.convolve(
                    active.astype(np.int32),
                    np.ones(10, dtype=np.int32),
                    mode="same",
                )
                >= 4
            )
            indices = np.flatnonzero(sustained)
            if not len(indices):
                raise ValueError("No sustained sound detected. Check the microphone and try again.")
            start = max(0, int(indices[0]) * window - SAMPLE_RATE // 2)
            end = min(len(data), (int(indices[-1]) + 1) * window + SAMPLE_RATE // 2)
            if end - start >= 3 * SAMPLE_RATE:
                trim_start = start / SAMPLE_RATE
                trim_end = (len(data) - end) / SAMPLE_RATE
                data = data[start:end]
            gain = 0.5 / max(float(np.max(np.abs(data))), 1e-10)
            if gain > 16:
                raise ValueError(
                    "The take is too quiet to normalize safely. Move closer and retry."
                )
            data = data * gain
            gain_db = float(20 * np.log10(gain))
        quality = audio_quality(data)
        if quality["reasons"]:
            raise ValueError(" ".join(quality["reasons"]))
        index = self.index
        prepared = source.with_name(f"sample-{index + 1:02}-{uuid.uuid4().hex[:12]}.wav")
        write_wav(prepared, data)
        prepared_digest = hashlib.sha256(prepared.read_bytes()).hexdigest()
        for sample in self.draft.samples:
            if (
                sample
                and hashlib.sha256(
                    self.workspace.inside(self.folder / sample.audio).read_bytes()
                ).hexdigest()
                == prepared_digest
            ):
                prepared.unlink()
                raise ValueError("This duplicates a saved recording. Read the current passage.")
        self.draft.samples[index] = Sample(
            audio=str(prepared.relative_to(self.folder)),
            text=self.draft.readings[index].text,
        )
        report = {
            "raw": source.name,
            "prepared": prepared.name,
            "quality": quality,
            "gain_db": gain_db,
            "trim_start_seconds": trim_start,
            "trim_end_seconds": trim_end,
            "transcript_confirmed": True,
            "notice": "Signal checks only. Listen for pronunciation, speaker identity and noise.",
        }
        save_json(prepared.with_suffix(".quality.json"), report)
        save_json(self.path, self.draft)
        save_json(
            self.folder / "samples.json",
            {
                "samples": [sample.model_dump() for sample in self.draft.samples if sample],
            },
        )
        return report

    def finish(self) -> Voice:
        if not self.finished:
            raise ValueError("Record and save every passage before enrolling this alias.")
        save_json(
            self.folder / "samples.json",
            {
                "samples": [sample.model_dump() for sample in self.draft.samples if sample],
            },
        )
        voice = enroll(
            self.workspace,
            self.alias,
            self.folder / "samples.json",
            consent=self.draft.consent,
            confined=True,
        )
        self.draft.completed = True
        save_json(self.path, self.draft)
        return voice


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

    def start(self, device: str, output: Path) -> None:
        if os.name != "nt":
            raise RuntimeError("Recording is supported on Windows only.")
        if self.process is not None and self.process.poll() is None:
            raise ValueError("A recording is already running.")
        if not device:
            raise ValueError("Select a microphone first.")
        if output.exists():
            raise ValueError("Refusing to overwrite an existing take.")
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
                    "-n",
                    "-f",
                    "dshow",
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
            process.communicate(
                input=b"q\n" if process.poll() is None else None,
                timeout=5,
            )
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
                encoding="utf-8",
                errors="replace",
            )[-2000:]
            raise RuntimeError("Microphone recording failed. " + detail)
        return self.output
