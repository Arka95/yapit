from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from yapit.media import decode_audio
from yapit.speech import MODEL_ID
from yapit.storage import Workspace, save_json
from yapit.voices import load_voice


def train(
    workspace: Workspace,
    name: str,
    upstream: Path,
    *,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-6,
) -> dict:
    if not 1 <= epochs <= 30 or not 1 <= batch_size <= 16 or not 1e-8 <= learning_rate <= 1e-3:
        raise ValueError("Invalid training parameters.")
    if platform.system() != "Linux":
        raise RuntimeError(
            "Upstream fine-tuning requires CUDA/bfloat16/FlashAttention. "
            "Run training on Linux or inside WSL2; inference can run separately on Windows."
        )
    try:
        import flash_attn  # noqa: F401
        import torch
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            'Install ".[train]" and a compatible FlashAttention build on the training machine.'
        ) from exc
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires an NVIDIA CUDA GPU with bfloat16 support.")
    upstream = upstream.resolve()
    training_dir = upstream / "finetuning"
    for filename in ["prepare_data.py", "sft_12hz.py", "dataset.py"]:
        if not (training_dir / filename).is_file():
            raise ValueError(f"Missing official upstream training file: {filename}")
    revision = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(upstream), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    if dirty.strip():
        raise ValueError("Use a clean, reviewed upstream checkout for reproducible training.")
    voice = load_voice(workspace, name)
    if len(voice.samples) < 10:
        raise ValueError(
            "Fine-tuning needs at least 10 curated distinct clips (project guardrail)."
        )
    voice_dir = workspace.voice_path(name).parent
    job = workspace.job()
    records = []
    for sample in voice.samples:
        audio = workspace.inside(voice_dir / sample.audio)
        records.append(
            {
                "audio": str(audio),
                "text": sample.text,
                "ref_audio": str(workspace.inside(voice_dir / voice.reference_audio)),
            }
        )
    # Keep a deterministic holdout, never used by the upstream optimization loop.
    ordered = sorted(
        records, key=lambda row: hashlib.sha256(Path(row["audio"]).read_bytes()).hexdigest()
    )
    heldout = max(1, len(ordered) // 10)
    splits = {"train": ordered[heldout:], "holdout": ordered[:heldout]}
    for split, rows in splits.items():
        (job / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    total = sum(len(decode_audio(Path(row["audio"]), max_seconds=20)) / 24000 for row in records)
    if total < 60:
        raise ValueError(
            "Fine-tuning needs at least 60 seconds; 5-15 clean minutes is recommended."
        )
    save_json(job / "status.json", {"status": "training", "upstream_commit": revision})
    try:
        # Upstream saves checkpoints by copying a local model directory, not a Hub identifier.
        local_model = snapshot_download(repo_id=MODEL_ID)
        provenance = {
            "upstream_commit": revision,
            "base_snapshot": local_model,
            "voice": name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "holdout_clips": heldout,
            "training_clips": len(splits["train"]),
            "automatic_quality_approval": False,
        }
        save_json(job / "training.json", provenance)
        commands = [
            [
                sys.executable,
                str(training_dir / "prepare_data.py"),
                "--device",
                "cuda:0",
                "--input_jsonl",
                str(job / "train.jsonl"),
                "--output_jsonl",
                str(job / "train_codes.jsonl"),
            ],
            [
                sys.executable,
                str(training_dir / "sft_12hz.py"),
                "--init_model_path",
                local_model,
                "--output_model_path",
                str(job / "checkpoints"),
                "--train_jsonl",
                str(job / "train_codes.jsonl"),
                "--num_epochs",
                str(epochs),
                "--batch_size",
                str(batch_size),
                "--lr",
                str(learning_rate),
                "--speaker_name",
                name,
            ],
        ]
        with (job / "training.log").open("w", encoding="utf-8") as log:
            for command in commands:
                subprocess.run(
                    command,
                    cwd=training_dir,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=24 * 3600,
                )
        checkpoints = [
            str(path)
            for path in sorted((job / "checkpoints").glob("checkpoint-epoch-*"))
            if (path / "model.safetensors").is_file() and (path / "config.json").is_file()
        ]
        if not checkpoints:
            raise RuntimeError("Training produced no usable checkpoint.")
        result = {
            "status": "trained_needs_listening_review",
            "job": str(job),
            "checkpoints": checkpoints,
            "next": "Listen to held-out phrases and run voice-use-checkpoint to select an epoch.",
        }
        save_json(job / "status.json", result)
        return result
    except Exception as exc:
        save_json(job / "status.json", {"status": "failed", "error": str(exc)})
        raise


def use_checkpoint(workspace: Workspace, name: str, checkpoint: Path) -> None:
    checkpoint = checkpoint.resolve()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("tts_model_type") != "custom_voice":
        raise ValueError("Select an upstream fine-tuned CustomVoice checkpoint, not a Base model.")
    speakers = config.get("talker_config", {}).get("spk_id", {})
    if name not in speakers or not (checkpoint / "model.safetensors").is_file():
        raise ValueError("Checkpoint is incomplete or does not contain this speaker name.")
    voice = load_voice(workspace, name)
    voice.checkpoint = str(checkpoint)
    voice.speaker = name
    save_json(workspace.voice_path(name), voice)
