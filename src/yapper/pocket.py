from __future__ import annotations

import gc
import logging
import threading
from pathlib import Path

import numpy as np

from yapit.common_utils.metrics import timed
from yapit.models import Voice
from yapit.yapper.backends import asset_stamps, normalize_audio, package_version


class PocketSpeech:
    def __init__(self, voice: Voice, voice_dir: Path, *, model: str = "english_2026-10"):
        self.source = model
        self.reference = (
            voice_dir / getattr(voice, "pocket_voice_state", voice.reference_audio)
        ).resolve()
        if not self.reference.is_relative_to(voice_dir.resolve().parent):
            raise ValueError("Pocket voice state/reference must stay inside the profile folder.")
        self.model = self.voice_state = None
        self.lock = threading.Lock()

    def cache_identity(self) -> dict:
        config = Path(self.source)
        return {
            "model": self.source,
            "assets": asset_stamps([config, self.reference]),
            "version": package_version("pocket-tts"),
        }

    def describe(self) -> dict:
        return {
            "backend": "pocket",
            "model": self.source,
            "voice_mode": "reference",
            "device": "cpu",
            "dtype": "sdk_default",
            "inference_backend": "pocket",
        }

    @timed
    def load(self) -> None:
        if self.model is not None:
            return
        logging.getLogger(__name__).info(
            "Loading Pocket speech model and preparing the voice; first use may take a while."
        )
        try:
            from pocket_tts import TTSModel
        except ImportError as exc:
            raise RuntimeError(
                'Install Pocket TTS dependencies: pip install -e ".[pocket]"'
            ) from exc
        if not self.reference.is_file():
            raise ValueError("Pocket voice state/reference file does not exist.")
        options = (
            {"config": self.source}
            if Path(self.source).suffix in {".yaml", ".yml"}
            or self.source.startswith(("https://", "hf://"))
            else {"language": self.source}
        )
        model = TTSModel.load_model(**options)
        state = model.get_state_for_audio_prompt(str(self.reference))
        self.model, self.voice_state = model, state

    def close(self) -> None:
        with self.lock:
            self.model = self.voice_state = None
            gc.collect()

    @timed
    def synthesize(self, text: str) -> np.ndarray:
        with self.lock:
            self.load()
            wave = self.model.generate_audio(self.voice_state, text)
            sample_rate = self.model.sample_rate
        return normalize_audio(wave.detach().cpu().numpy(), sample_rate)
