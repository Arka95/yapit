"""CosyVoice SDK entry point: standard-library imports until the load request."""

import os
import sys
from pathlib import Path

MAX_SECONDS = 45


class CosyVoiceSession:
    def __init__(self):
        self.model = None

    def load(self, request):
        source = Path(request["source"])
        sys.path[:0] = [str(source), str(source / "third_party" / "Matcha-TTS")]
        device = request["device"]
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif device.startswith("cuda:"):
            index = device.removeprefix("cuda:")
            if not index.isdigit():
                raise ValueError("Use auto, cpu or cuda:N for CosyVoice.")
            os.environ["CUDA_VISIBLE_DEVICES"] = index
        elif device not in ("auto", "cuda"):
            raise ValueError("Use auto, cpu or cuda:N for CosyVoice.")

        import torch
        from cosyvoice.cli.cosyvoice import AutoModel

        cuda = torch.cuda.is_available()
        if device.startswith("cuda") and not cuda:
            raise RuntimeError("Explicit CosyVoice CUDA mode is unavailable.")
        model = AutoModel(
            model_dir=request["model"],
            fp16=cuda,
            load_trt=False,
            load_vllm=False,
        )
        components = {name: getattr(model.model, name) for name in ("llm", "flow", "hift")}
        devices = {next(component.parameters()).device.type for component in components.values()}
        if devices != {"cuda" if cuda else "cpu"}:
            raise RuntimeError("CosyVoice components were not loaded on the selected device.")
        dtypes = {
            name: str(next(component.parameters()).dtype).removeprefix("torch.")
            for name, component in components.items()
        }
        reference_text = request["reference_text"]
        if type(model).__name__ == "CosyVoice3":
            reference_text = "You are a helpful assistant.<|endofprompt|>" + reference_text
        if not model.add_zero_shot_spk(reference_text, request["reference"], "yapit"):
            raise RuntimeError("CosyVoice could not prepare the enrolled voice reference.")
        self.model = model
        self.output_root = Path(request["output_root"]).resolve()
        return {
            "variant": type(model).__name__,
            "device": "cuda:0" if cuda else "cpu",
            "dtype": next(iter(dtypes.values())) if len(set(dtypes.values())) == 1 else "mixed",
            "component_dtypes": dtypes,
            "torch_version": torch.__version__,
            "sample_rate": model.sample_rate,
            "runtime_pid": os.getpid(),
        }

    def synthesize(self, request):
        import numpy as np
        import soundfile as sf

        if self.model is None:
            raise RuntimeError("Load the model before synthesis.")
        output = Path(request["output"]).resolve()
        if output.parent != self.output_root or output.suffix != ".wav":
            raise ValueError("Runtime output must stay inside its private audio directory.")
        text = request["text"]
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Supply 1-4000 characters of speech text.")
        chunks = []
        samples = 0
        for result in self.model.inference_zero_shot(
            text, "", "", zero_shot_spk_id="yapit", stream=False, speed=1.0, text_frontend=False
        ):
            wave = result["tts_speech"].detach().float().cpu().numpy().reshape(-1)
            samples += len(wave)
            if samples / self.model.sample_rate > MAX_SECONDS:
                raise ValueError("A beat exceeded 45 seconds; split the script into shorter beats.")
            chunks.append(wave)
        if not chunks:
            raise RuntimeError("CosyVoice returned no speech chunks.")
        wave = np.concatenate(chunks)
        if not len(wave) or not np.isfinite(wave).all() or np.max(np.abs(wave)) < 1e-5:
            raise ValueError("CosyVoice produced empty, invalid or silent audio.")
        sf.write(output, wave, self.model.sample_rate, subtype="FLOAT")
        return {"sample_rate": self.model.sample_rate}


if __name__ == "__main__":
    from isolated_runtime import serve

    serve(CosyVoiceSession())
