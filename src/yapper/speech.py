from __future__ import annotations

import gc
import logging
import os
import shutil
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType

import numpy as np

from yapit.common_utils.metrics import timed
from yapit.common_utils.modelutils import load_model
from yapit.constants import DEFAULT_SPEECH_MODEL
from yapit.models import Voice
from yapit.yapper.backends import SpeechBackend as SpeechBackend
from yapit.yapper.backends import (
    asset_stamps,
    normalize_audio,
    package_version,
)

MODEL_ID = DEFAULT_SPEECH_MODEL


class QwenSpeech:
    def __init__(self, voice: Voice, voice_dir: Path, *, model: str | None = None):
        self.voice = voice
        self.voice_dir = voice_dir
        self.model = None
        self.prompt = None
        self.lock = threading.Lock()
        self.accelerator = None
        self.source = model or voice.checkpoint or os.environ.get("YAPIT_MODEL", MODEL_ID)
        self.mode = "custom_voice" if voice.checkpoint and model is None else "base"
        self.explicit_model = model is not None
        self.device = "unloaded"
        self.dtype = "unloaded"

    def cache_identity(self) -> dict:
        source = Path(self.source)
        paths = []
        if source.is_dir():
            paths.extend(source.glob("model*.safetensors"))
            paths.extend((source / "speech_tokenizer").glob("model*.safetensors"))
            paths.extend(source / name for name in ("config.json", "generation_config.json"))
        if self.mode == "base" or self.explicit_model:
            paths.append(self.voice_dir / self.voice.reference_audio)
        return {
            "model": self.source,
            "speaker": self.voice.speaker or self.voice.name,
            "reference_text": (
                self.voice.reference_text if self.mode == "base" or self.explicit_model else None
            ),
            "assets": asset_stamps(paths),
            "version": package_version("qwen-tts"),
        }

    def describe(self) -> dict:
        return {
            "backend": "qwen",
            "model": self.source,
            "voice_mode": "fine_tuned" if self.mode == "custom_voice" else "reference",
            "device": self.device,
            "dtype": self.dtype,
            "inference_backend": self.inference_backend,
        }

    @property
    def inference_backend(self) -> str:
        return "cuda_graphs" if self.accelerator and self.accelerator.active else "sdpa"

    @timed
    def load(self) -> None:
        if self.model is not None:
            return
        logging.getLogger(__name__).info(
            "Loading Qwen speech model and preparing the voice; first use may take a while."
        )
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
            if device == "cpu":
                logging.getLogger(__name__).warning(
                    "No CUDA GPU is available; defaulting to CPU narration."
                )
        dtype = (
            torch.float32
            if device == "cpu"
            else torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
        with redirect_stdout(sys.stderr):
            model = load_model(
                Qwen3TTSModel.from_pretrained,
                self.source,
                device_map=device,
                dtype=dtype,
                attn_implementation="sdpa",
            )
            mode = getattr(getattr(model, "model", None), "tts_model_type", self.mode)
            if mode not in {"base", "custom_voice"}:
                raise ValueError("This Qwen adapter supports Base and CustomVoice models.")
            prompt = None
            if mode == "base":
                logging.getLogger(__name__).info("Preparing Qwen voice reference.")
                reference = (self.voice_dir / self.voice.reference_audio).resolve()
                if (
                    not reference.is_relative_to(self.voice_dir.resolve())
                    or not reference.is_file()
                ):
                    raise ValueError("Invalid local reference path in voice profile.")
                prompt = model.create_voice_clone_prompt(
                    ref_audio=str(reference),
                    ref_text=self.voice.reference_text,
                    x_vector_only_mode=False,
                )
            self.model, self.prompt, self.mode = model, prompt, mode
            self._enable_acceleration(torch, device)
        self.device, self.dtype = device, str(dtype)

    def close(self) -> None:
        with self.lock:
            self.model = self.prompt = self.accelerator = None
            self.device = self.dtype = "unloaded"
            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _enable_acceleration(self, torch: ModuleType, device: str) -> None:
        if device == "cpu" or os.environ.get("YAPIT_CUDA_GRAPHS", "1") == "0":
            return
        from yapit.yapper.inference import CudaCodePredictor

        talker = self.model.model.talker
        self.accelerator = CudaCodePredictor(
            talker.code_predictor, torch, talker.config.num_code_groups
        )
        talker.code_predictor.generate = self.accelerator.generate

    @timed
    def synthesize(self, text: str) -> np.ndarray:
        with self.lock, redirect_stdout(sys.stderr):
            self.load()
            kwargs = {"text": text, "language": "English", "max_new_tokens": 2048}
            if self.mode == "custom_voice":
                waves, sample_rate = self.model.generate_custom_voice(
                    **kwargs,
                    speaker=self.voice.speaker or self.voice.name,
                )
            else:
                waves, sample_rate = self.model.generate_voice_clone(
                    **kwargs,
                    voice_clone_prompt=self.prompt,
                )
        return normalize_audio(waves[0], sample_rate)
