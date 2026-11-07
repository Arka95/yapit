from __future__ import annotations

import os
import shutil
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path
from typing import Protocol

import numpy as np

from yapit.media import SAMPLE_RATE, resample
from yapit.models import Voice

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


class SpeechBackend(Protocol):
    def synthesize(self, text: str) -> np.ndarray: ...


class QwenSpeech:
    def __init__(self, voice: Voice, voice_dir: Path):
        self.voice = voice
        self.voice_dir = voice_dir
        self.model = None
        self.prompt = None
        self.lock = threading.Lock()

    def _load(self) -> None:
        if self.model is not None:
            return
        scripts = Path(sys.executable).parent
        sox_binary = scripts / ("sox.exe" if os.name == "nt" else "sox")
        if shutil.which("sox") is None and sox_binary.is_file():
            os.environ["PATH"] = str(scripts) + os.pathsep + os.environ.get("PATH", "")
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError(
                'Install local speech dependencies: pip install -e ".[voice]"'
            ) from exc
        device = os.environ.get("YAPIT_DEVICE", "auto")
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = (
            torch.float32
            if device == "cpu"
            else torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
        with redirect_stdout(sys.stderr):
            self.model = Qwen3TTSModel.from_pretrained(
                self.voice.checkpoint or os.environ.get("YAPIT_MODEL", MODEL_ID),
                device_map=device,
                dtype=dtype,
                attn_implementation="sdpa",
            )
            if self.voice.checkpoint is None:
                reference = (self.voice_dir / self.voice.reference_audio).resolve()
                if (
                    not reference.is_relative_to(self.voice_dir.resolve())
                    or not reference.is_file()
                ):
                    raise ValueError("Invalid local reference path in voice profile.")
                self.prompt = self.model.create_voice_clone_prompt(
                    ref_audio=str(reference),
                    ref_text=self.voice.reference_text,
                    x_vector_only_mode=False,
                )

    def synthesize(self, text: str) -> np.ndarray:
        with self.lock, redirect_stdout(sys.stderr):
            self._load()
            kwargs = {"text": text, "language": "English", "max_new_tokens": 2048}
            if self.voice.checkpoint:
                waves, sample_rate = self.model.generate_custom_voice(
                    **kwargs,
                    speaker=self.voice.speaker or self.voice.name,
                )
            else:
                waves, sample_rate = self.model.generate_voice_clone(
                    **kwargs,
                    voice_clone_prompt=self.prompt,
                )
        data = np.asarray(waves[0], dtype=np.float32).reshape(-1)
        if not len(data) or not np.isfinite(data).all() or np.max(np.abs(data)) < 1e-5:
            raise ValueError("Speech model produced empty, invalid or silent audio.")
        data = resample(data, sample_rate)
        if len(data) / SAMPLE_RATE > 45:
            raise ValueError("A beat exceeded 45 seconds; split the script into shorter beats.")
        return data
