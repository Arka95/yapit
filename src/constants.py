import os
from pathlib import Path

DEFAULT_PROFILE = "arka"
DATA_FILENAME_TEMPLATE = "{profile}.{name}{suffix}"
AUDIO_SAMPLE_NAME_TEMPLATE = "audio_sample_{number}"
GENERATED_VIDEO_DIRECTORY = "generated_video"
DEFAULT_MODELS_DIRECTORY = "models"
MODEL_CACHE_DIRECTORY = "hub"
DEFAULT_SPEECH_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_ASR_MODEL = "small.en"


def models_directory() -> Path:
    directory = Path(os.environ.get("YAPIT_MODELS_DIR") or DEFAULT_MODELS_DIRECTORY).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory
