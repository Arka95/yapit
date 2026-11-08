from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from yapit.common_utils.metrics import timed
from yapit.constants import models_directory
from yapit.models import Voice
from yapit.yapper.backends import asset_stamps, normalize_audio
from yapit.yapper.isolated import RuntimeProcess

DEFAULT_MODEL = "FunAudioLLM/CosyVoice2-0.5B"
MODEL_FOLDERS = {
    DEFAULT_MODEL: "cosyvoice2-0.5b",
    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512": "cosyvoice3-2512",
}


class CosyVoiceSpeech:
    def __init__(self, voice: Voice, voice_dir: Path, *, model: str = DEFAULT_MODEL):
        self.model_id = model
        self.directory = voice_dir.resolve()
        self.reference = (self.directory / voice.reference_audio).resolve()
        if not self.reference.is_relative_to(self.directory):
            raise ValueError("CosyVoice reference must stay inside the profile audio folder.")
        self.reference_text = voice.reference_text
        root = models_directory()
        self.model_dir = (
            root / MODEL_FOLDERS[model] if model in MODEL_FOLDERS else Path(model)
        ).resolve()
        self.source = Path(
            os.environ.get("YAPIT_COSYVOICE_SOURCE", str(root / "cosyvoice_source"))
        ).resolve()
        executable = Path(".venv-cosyvoice") / (
            Path("Scripts") / "python.exe" if os.name == "nt" else Path("bin") / "python"
        )
        self.python = Path(os.environ.get("YAPIT_COSYVOICE_PYTHON", str(executable))).resolve()
        self.device = os.environ.get("YAPIT_DEVICE", "auto")
        self.runtime = self.temporary = None
        self.info = {}
        self.lock = threading.RLock()

    def cache_identity(self) -> dict:
        paths = [
            self.reference,
            self.python,
            Path(__file__),
            Path(__file__).with_name("cosyvoice_runtime.py"),
            Path(__file__).with_name("isolated_runtime.py"),
            Path(__file__).with_name("isolated.py"),
        ]
        for pattern in ("*.pt", "*.onnx", "*.yaml", "CosyVoice-BlankEN/*"):
            paths.extend(self.model_dir.glob(pattern))
        paths.extend(self.source.glob("cosyvoice/**/*.py"))
        paths.extend(self.source.glob("third_party/Matcha-TTS/matcha/**/*.py"))
        paths.extend(
            (self.python.parent.parent / "Lib" / "site-packages").glob("*.dist-info/METADATA")
        )
        return {
            "model": self.model_id,
            "reference_text": self.reference_text,
            "device": self.device,
            "assets": asset_stamps(paths),
        }

    def describe(self) -> dict:
        return {
            **self.info,
            "backend": "cosyvoice",
            "model": self.model_id,
            "voice_mode": "reference",
            "device": self.info.get("device", "unloaded"),
            "dtype": self.info.get("dtype", "unloaded"),
            "inference_backend": "isolated_cosyvoice",
        }

    @timed
    def load(self) -> None:
        with self.lock:
            if self.runtime is not None:
                return
            if (
                not self.python.is_file()
                or not (self.source / "cosyvoice" / "cli" / "cosyvoice.py").is_file()
            ):
                raise RuntimeError(
                    "Provision the isolated CosyVoice runtime/source; configure "
                    "YAPIT_COSYVOICE_PYTHON and YAPIT_COSYVOICE_SOURCE if needed."
                )
            if not any(
                (self.model_dir / name).is_file() for name in ("cosyvoice2.yaml", "cosyvoice3.yaml")
            ):
                raise ValueError(
                    "Provision a local CosyVoice2/3 model directory before loading it."
                )
            if not self.reference.is_file():
                raise ValueError("CosyVoice reference audio does not exist.")
            self.temporary = tempfile.TemporaryDirectory(prefix="yapit-cosyvoice-")
            try:
                self.runtime = RuntimeProcess(
                    [str(self.python), str(Path(__file__).with_name("cosyvoice_runtime.py"))],
                    cwd=self.source,
                    env=os.environ.copy(),
                )
                self.info = self.runtime.request(
                    {
                        "action": "load",
                        "source": str(self.source),
                        "model": str(self.model_dir),
                        "reference": str(self.reference),
                        "reference_text": self.reference_text,
                        "device": self.device,
                        "output_root": self.temporary.name,
                    },
                    timeout=300,
                )
            except Exception:
                self.close()
                raise

    def close(self) -> None:
        with self.lock:
            try:
                if self.runtime is not None:
                    self.runtime.close()
            finally:
                self.runtime = None
                self.info = {}
                if self.temporary is not None:
                    self.temporary.cleanup()
                    self.temporary = None

    @timed
    def synthesize(self, text: str) -> np.ndarray:
        with self.lock:
            self.load()
            output = Path(self.temporary.name) / "speech.wav"
            try:
                response = self.runtime.request(
                    {"action": "synthesize", "text": text, "output": str(output)}
                )
                wave, sample_rate = sf.read(output, dtype="float32")
                if sample_rate != response.get("sample_rate"):
                    raise RuntimeError("Isolated audio sample rate does not match its metadata.")
                return normalize_audio(wave, sample_rate)
            except Exception:
                self.close()
                raise
            finally:
                output.unlink(missing_ok=True)
