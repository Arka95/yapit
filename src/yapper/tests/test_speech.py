import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from yapit.common_utils.mediautils import write_wav
from yapit.enrollment import load_voice
from yapit.enrollment.styles import enroll_style, select_voice
from yapit.models import DeliveryStyle
from yapit.yapper.speech import QwenSpeech

from .conftest import tone


@pytest.mark.parametrize("trained,styled", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_qwen_uses_reference_or_checkpoint_without_cloud(
    enrolled, monkeypatch, trained, styled, tmp_path, device, caplog
):
    scripts = tmp_path / "venv-bin"
    scripts.mkdir()
    (scripts / ("sox.exe" if os.name == "nt" else "sox")).write_bytes(b"test executable")
    monkeypatch.setattr(sys, "executable", str(scripts / "python.exe"))
    monkeypatch.setattr("yapit.yapper.speech.shutil.which", lambda name: None)
    calls = []
    model = SimpleNamespace(
        create_voice_clone_prompt=lambda **kwargs: calls.append(("prompt", kwargs)) or ["cached"],
        generate_voice_clone=lambda **kwargs: calls.append(("clone", kwargs)) or ([tone(1)], 24000),
        generate_custom_voice=lambda **kwargs: (
            calls.append(("trained", kwargs)) or ([tone(1)], 24000)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float32="float32",
            bfloat16="bfloat16",
            float16="float16",
            cuda=SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        SimpleNamespace(
            Qwen3TTSModel=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: (
                    calls.append(("load", (args, kwargs))) or model
                )
            )
        ),
    )
    monkeypatch.setenv("YAPIT_DEVICE", device)
    voice = load_voice(enrolled.data, "test-voice")
    voice_dir = enrolled.voice_audio_dir(voice.name)
    if styled:
        source = tmp_path / "lively.wav"
        write_wav(source, tone(4, 260))
        enroll_style(
            enrolled.data,
            voice.name,
            DeliveryStyle.lively_vlog,
            source,
            "What a lovely surprise!",
            True,
        )
        voice, voice_dir = select_voice(enrolled.data, voice.name, DeliveryStyle.lively_vlog)
    if trained:
        voice.checkpoint = "local-checkpoint"
        voice.speaker = voice.name
    backend = QwenSpeech(voice, voice_dir)
    assert np.isfinite(backend.synthesize("First sentence.")).all()
    backend.synthesize("Second sentence.")
    assert sum((kind == "load" for kind, _ in calls)) == 1
    loaded = next((options for kind, options in calls if kind == "load"))
    assert loaded[1]["cache_dir"] == str(tmp_path / "models" / "hub")
    assert loaded[1]["device_map"] == "cpu"
    warnings = [record for record in caplog.records if "CPU narration" in record.message]
    assert len(warnings) == (1 if device == "auto" else 0)
    assert sum((kind == "prompt" for kind, _ in calls)) == (0 if trained else 1)
    assert sum((kind == ("trained" if trained else "clone") for kind, _ in calls)) == 2
    if styled:
        prompt = next((kwargs for kind, kwargs in calls if kind == "prompt"))
        assert prompt["ref_text"] == "What a lovely surprise!"
        assert prompt["ref_audio"] == str(voice_dir / voice.reference_audio)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(scripts)
