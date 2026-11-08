import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from yapit.yapper.inference import CudaCodePredictor
from yapit.yapper.speech import QwenSpeech


@pytest.fixture
def predictor():
    return SimpleNamespace(
        generate=Mock(return_value="sdk-result"),
        generation_config=SimpleNamespace(
            to_diff_dict=lambda: {},
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
        ),
    )


def generation_args():
    return {
        "inputs_embeds": SimpleNamespace(
            is_cuda=True, ndim=3, shape=(1, 2, 8), dtype="bf16", device="cuda"
        ),
        "max_new_tokens": 15,
        "return_dict_in_generate": True,
    }


@pytest.mark.parametrize(
    "unsupported", ["cpu", "batch", "training", "sampling_processor", "configuration"]
)
def test_unsupported_requests_preserve_sdk_generation(predictor, unsupported):
    kwargs = generation_args()
    if unsupported == "cpu":
        kwargs["inputs_embeds"].is_cuda = False
    elif unsupported == "batch":
        kwargs["inputs_embeds"].shape = (2, 2, 8)
    elif unsupported == "training":
        predictor.training = True
    elif unsupported == "sampling_processor":
        kwargs["logits_processor"] = object()
    else:
        predictor.generation_config.to_diff_dict = lambda: {"repetition_penalty": 1.2}
    adapter = CudaCodePredictor(predictor, SimpleNamespace(), 16)

    assert adapter.generate(**kwargs) == "sdk-result"
    predictor.generate.assert_called_once_with(**kwargs)


def test_failed_capture_falls_back_without_retries(predictor, monkeypatch, caplog):
    adapter = CudaCodePredictor(predictor, SimpleNamespace(inference_mode=nullcontext), 16)
    capture = Mock(side_effect=RuntimeError("capture unsupported"))
    monkeypatch.setattr(adapter, "_capture", capture)

    assert adapter.generate(**generation_args()) == "sdk-result"
    assert adapter.generate(**generation_args()) == "sdk-result"
    assert capture.call_count == 1
    assert predictor.generate.call_count == 2
    assert not adapter.active
    assert "using SDK generation" in caplog.text


def test_backend_status_tracks_graph_replay_then_sdk_fallback(predictor, monkeypatch):
    adapter = CudaCodePredictor(predictor, SimpleNamespace(inference_mode=nullcontext), 16)

    def capture(*args):
        adapter.inputs = SimpleNamespace(copy_=Mock())
        adapter.graph = SimpleNamespace(replay=Mock())
        adapter.tokens = SimpleNamespace(clone=lambda: "graph-result")
        adapter.output_type = SimpleNamespace

    monkeypatch.setattr(adapter, "_capture", capture)
    assert adapter.generate(**generation_args()).sequences == "graph-result"
    assert adapter.active
    assert adapter.generate(**generation_args(), logits_processor=object()) == "sdk-result"
    assert not adapter.active


@pytest.mark.parametrize("device,enabled", [("cuda", True), ("cuda", False), ("cpu", True)])
def test_gpu_optimization_is_default_and_can_be_disabled(tmp_path, monkeypatch, device, enabled):
    torch = SimpleNamespace(
        float32="fp32",
        bfloat16="bf16",
        float16="fp16",
        cuda=SimpleNamespace(is_bf16_supported=lambda: True),
    )
    predictor = SimpleNamespace(generate=Mock())
    model = SimpleNamespace(
        model=SimpleNamespace(
            talker=SimpleNamespace(
                code_predictor=predictor, config=SimpleNamespace(num_code_groups=7)
            )
        ),
        generate_custom_voice=Mock(return_value=([np.full(24, 0.1, dtype=np.float32)], 24000)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=Mock(return_value=model))),
    )
    monkeypatch.setattr("yapit.yapper.speech.shutil.which", lambda name: "sox")
    monkeypatch.setenv("YAPIT_DEVICE", device)
    monkeypatch.delenv("YAPIT_CUDA_GRAPHS", raising=False)
    if not enabled:
        monkeypatch.setenv("YAPIT_CUDA_GRAPHS", "0")
    optimized = SimpleNamespace(active=True, generate=Mock())
    factory = Mock(return_value=optimized)
    monkeypatch.setattr("yapit.yapper.inference.CudaCodePredictor", factory)
    voice = SimpleNamespace(checkpoint="local-checkpoint", name="arka", speaker="arka")
    backend = QwenSpeech(voice, tmp_path)

    assert np.isfinite(backend.synthesize("A short sentence.")).all()
    if device == "cuda" and enabled:
        factory.assert_called_once_with(predictor, torch, 7)
        assert predictor.generate is optimized.generate
        assert backend.inference_backend == "cuda_graphs"
    else:
        factory.assert_not_called()
        assert backend.inference_backend == "sdpa"
