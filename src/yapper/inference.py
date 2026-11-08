from __future__ import annotations

import logging
from types import ModuleType
from typing import TYPE_CHECKING, Any

from yapit.common_utils.metrics import timed

if TYPE_CHECKING:
    from torch import Tensor

GENERATION_OPTIONS = {
    "inputs_embeds",
    "max_new_tokens",
    "do_sample",
    "temperature",
    "top_k",
    "top_p",
    "output_hidden_states",
    "return_dict_in_generate",
}
SAMPLING_OPTIONS = ("do_sample", "temperature", "top_k", "top_p")
CONFIG_OPTIONS = GENERATION_OPTIONS | {"_from_model_config", "transformers_version"}


class CudaCodePredictor:
    """Replay the fixed-length codebook loop without per-token Python dispatch."""

    def __init__(self, predictor: Any, torch: ModuleType, code_groups: int):
        self.predictor = predictor
        self.torch = torch
        self.code_tokens = code_groups - 1
        self.original = predictor.generate
        self.graph = None
        self.key = None
        self.disabled = False
        self.used_graph = False

    @property
    def active(self) -> bool:
        return self.used_graph

    def _supported(self, args: tuple, kwargs: dict) -> bool:
        inputs = kwargs.get("inputs_embeds")
        config = self.predictor.generation_config
        return (
            not args
            and not kwargs.keys() - GENERATION_OPTIONS
            and self.code_tokens > 0
            and not getattr(self.predictor, "training", False)
            and inputs is not None
            and inputs.is_cuda
            and inputs.ndim == 3
            and tuple(inputs.shape[:2]) == (1, 2)
            and kwargs.get("max_new_tokens") == self.code_tokens
            and kwargs.get("return_dict_in_generate") is True
            and not config.to_diff_dict().keys() - CONFIG_OPTIONS
        )

    def generate(self, *args, **kwargs) -> Any:
        self.used_graph = False
        if self.disabled or not self._supported(args, kwargs):
            return self.original(*args, **kwargs)
        inputs = kwargs["inputs_embeds"]
        settings = tuple(
            kwargs[name]
            if kwargs.get(name) is not None
            else getattr(self.predictor.generation_config, name)
            for name in SAMPLING_OPTIONS
        )
        key = (settings, inputs.dtype, inputs.device, tuple(inputs.shape))
        with self.torch.inference_mode():
            if key != self.key:
                try:
                    self._capture(inputs, settings)
                except (RuntimeError, TypeError, AttributeError) as exc:
                    self.disabled = True
                    logging.getLogger(__name__).warning(
                        "CUDA graph capture unavailable (%s); using SDK generation.",
                        type(exc).__name__,
                    )
                    return self.original(*args, **kwargs)
                self.key = key
            self.inputs.copy_(inputs)
            self.graph.replay()
            self.used_graph = True
            return self.output_type(sequences=self.tokens.clone())

    @timed
    def _capture(self, inputs: Tensor, settings: tuple) -> None:
        from transformers.generation.logits_process import (
            LogitsProcessorList,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )
        from transformers.generation.utils import GenerateDecoderOnlyOutput

        sample, temperature, top_k, top_p = settings
        self.sample = sample
        self.processors = LogitsProcessorList()
        if sample:
            if temperature != 1:
                self.processors.append(TemperatureLogitsWarper(temperature))
            if top_k:
                self.processors.append(TopKLogitsWarper(top_k))
            if top_p != 1:
                self.processors.append(TopPLogitsWarper(top_p))
        self.output_type = GenerateDecoderOnlyOutput
        self.inputs = inputs.detach().clone()
        self.graph = self.torch.cuda.CUDAGraph()
        stream = self.torch.cuda.Stream(device=inputs.device)
        current = self.torch.cuda.current_stream(inputs.device)
        state = self.torch.cuda.get_rng_state(inputs.device)
        try:
            stream.wait_stream(current)
            with self.torch.cuda.stream(stream):
                for _ in range(2):
                    self._decode()
            current.wait_stream(stream)
            with self.torch.cuda.graph(self.graph, stream=stream):
                self.tokens = self._decode()
        finally:
            self.torch.cuda.set_rng_state(state, inputs.device)
        logging.getLogger(__name__).info("Enabled CUDA-graph codebook generation.")

    def _decode(self) -> Tensor:
        torch = self.torch
        sequence = torch.empty((1, 0), dtype=torch.long, device=self.inputs.device)
        cache = None
        token = None
        for step in range(self.code_tokens):
            output = self.predictor(
                inputs_embeds=self.inputs if step == 0 else None,
                input_ids=token,
                past_key_values=cache,
                generation_steps=step,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            scores = output.logits[:, -1, :].clone().float()
            if self.sample:
                scores = self.processors(sequence, scores)
                token = torch.multinomial(torch.softmax(scores, dim=-1), 1)
            else:
                token = scores.argmax(dim=-1, keepdim=True)
            sequence = torch.cat((sequence, token), dim=1)
            cache = output.past_key_values
        return sequence
