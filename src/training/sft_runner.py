from __future__ import annotations

import runpy
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace


def run_sft(script: Path, logging_dir: Path, arguments: list[str]) -> None:
    original_argv, original_path = sys.argv, sys.path.copy()
    hooks = []
    try:
        sys.path.insert(0, str(script.parent))
        namespace = runpy.run_path(str(script))
        train = namespace["train"]
        # Supply the missing logging directory without editing the reviewed checkout.
        train.__globals__["Accelerator"] = partial(
            train.__globals__["Accelerator"], project_dir=str(logging_dir)
        )
        model_class = train.__globals__["Qwen3TTSModel"]

        def load_model(*args, **kwargs):
            result = model_class.from_pretrained(*args, **kwargs)
            talker = result.model.talker
            text = talker.model.text_embedding
            codec = talker.model.codec_embedding
            if text.embedding_dim != codec.embedding_dim:
                # Upstream omits the projection required by the smaller Base architecture.
                hooks.append(
                    text.register_forward_hook(
                        lambda module, inputs, output: talker.text_projection(output)
                    )
                )
            return result

        train.__globals__["Qwen3TTSModel"] = SimpleNamespace(from_pretrained=load_model)
        sys.argv = [str(script), *arguments]
        train()
    finally:
        for hook in hooks:
            hook.remove()
        sys.argv = original_argv
        sys.path[:] = original_path


if __name__ == "__main__":
    run_sft(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
