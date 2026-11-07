"""Trial CosyVoice3 in its own interpreter without changing YapIt's default backend."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
INSTRUCTION = (
    "You are a helpful assistant. Speak in a warm, engaged, lightly curious "
    "conversational tone for a travel vlog. Use natural emphasis, not theatrical "
    "acting. Keep a normal speaking rate and short natural sentence pauses."
    "<|endofprompt|>"
)


def worker(request_path: Path) -> None:
    import numpy as np
    import soundfile as sf
    import torch

    request = json.loads(request_path.read_text(encoding="utf-8"))
    source = ROOT / "models" / "cosyvoice_source"
    sys.path[:0] = [str(source), str(source / "third_party" / "Matcha-TTS")]
    from cosyvoice.cli.cosyvoice import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("CosyVoice trial requires the connected NVIDIA GPU.")
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = AutoModel(
        model_dir=str(ROOT / "models" / "cosyvoice3-2512"),
        fp16=True,
        load_trt=False,
        load_vllm=False,
    )
    for component in (model.model.llm, model.model.flow, model.model.hift):
        if next(component.parameters()).device.type != "cuda":
            raise RuntimeError("A neural synthesis component was not loaded on CUDA.")
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    chunks = []
    for result in model.inference_instruct2(
        request["text"],
        request["instruction"],
        request["reference"],
        stream=False,
        speed=1.0,
        text_frontend=False,
    ):
        chunks.append(result["tts_speech"].detach().float().cpu().numpy().reshape(-1))
    if not chunks:
        raise RuntimeError("CosyVoice returned no audio.")
    wave = np.concatenate(chunks)
    seconds = len(wave) / model.sample_rate
    if not np.isfinite(wave).all() or np.max(np.abs(wave)) < 1e-5 or not 5 <= seconds <= 190:
        raise RuntimeError("CosyVoice returned invalid, silent, or out-of-bounds audio.")
    synthesis_seconds = time.perf_counter() - started
    sf.write(request["raw"], wave, model.sample_rate, subtype="PCM_16")
    metrics = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "precision": "upstream fp16 mode",
        "model_load_seconds": load_seconds,
        "synthesis_seconds": synthesis_seconds,
        "audio_seconds": seconds,
        "real_time_factor": synthesis_seconds / seconds,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "onnx_providers": model.frontend.speech_tokenizer_session.get_providers(),
    }
    Path(request["metrics"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def trial(script_path: Path, profile: str, instruction: str) -> None:
    from yapit.cli import workspace
    from yapit.media import SAMPLE_RATE, decode_audio, encode_mp3, write_wav
    from yapit.scripts import load_script
    from yapit.storage import save_json
    from yapit.voices import select_voice

    root = workspace()
    script = load_script(script_path)
    voice, voice_dir = select_voice(root, profile)
    if voice.checkpoint:
        raise ValueError("The CosyVoice trial uses a reference, not a Qwen checkpoint.")
    reference = (voice_dir / voice.reference_audio).resolve()
    if not reference.is_relative_to(voice_dir.resolve()) or not reference.is_file():
        raise ValueError("Invalid local voice reference.")
    python = ROOT / ".venv-cosyvoice" / "Scripts" / "python.exe"
    if not python.is_file():
        raise RuntimeError("Provision the separate CosyVoice Python 3.10 runtime first.")
    output = root.output(script.title, profile)
    status_path = output.path("status.json")
    raw = output.path("raw.wav", output.audio)
    wav = output.path("wav", output.audio)
    mp3 = output.path("mp3", output.audio)
    metrics_path = output.path("metrics.json")
    request_path = output.path("cosyvoice.request.json")
    request = {
        "text": " ".join(beat.text for beat in script.beats),
        "instruction": instruction,
        "reference": str(reference),
        "raw": str(raw),
        "metrics": str(metrics_path),
    }
    save_json(request_path, request)
    save_json(status_path, {"status": "running", "name": output.name})
    try:
        environment = os.environ.copy()
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
        subprocess.run(
            [str(python), str(Path(__file__).resolve()), "--worker", str(request_path)],
            cwd=ROOT,
            env=environment,
            check=True,
            timeout=1800,
        )
        wave = decode_audio(raw)
        write_wav(wav, wave)
        encode_mp3(wav, mp3)
        encoded = decode_audio(mp3)
        if abs(len(encoded) / SAMPLE_RATE - len(wave) / SAMPLE_RATE) > 5:
            raise ValueError("MP3 encoding changed the duration beyond tolerance.")
        generated_script = script.model_copy(
            update={"beats": [beat.model_copy(update={"pause_ms": 0}) for beat in script.beats]}
        )
        saved_script = output.path("script.json", output.transcripts)
        save_json(saved_script, generated_script)
        source_revision = subprocess.check_output(
            ["git", "-C", str(ROOT / "models" / "cosyvoice_source"), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        provenance = output.path("provenance.json")
        save_json(
            provenance,
            {
                "backend": "cosyvoice3",
                "model": MODEL_ID,
                "source_revision": source_revision,
                "profile": profile,
                "reference_audio": str(reference),
                "instruction": instruction,
                "speaking_speed": 1.0,
                "pause_policy": "model punctuation only; no inserted source-timing pauses",
                "synthetic_audio": True,
                "review_required": True,
                "emotion_adherence": "requested, not yet listening-validated",
            },
        )
        result = {
            "status": "complete",
            "name": output.name,
            "wav": str(wav),
            "audio": str(mp3),
            "script": str(saved_script),
            "provenance": str(provenance),
            "metrics": str(metrics_path),
            "seconds": len(encoded) / SAMPLE_RATE,
            "review_required": True,
        }
        save_json(status_path, result)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        save_json(status_path, {"status": "failed", "name": output.name, "error": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--script", type=Path)
    parser.add_argument("--profile", default="arka")
    parser.add_argument("--instruction", default=INSTRUCTION)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.script:
        trial(args.script, args.profile, args.instruction)
    else:
        parser.error("Provide --script.")
