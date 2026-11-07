from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from yapit.common_utils.fileutils import file_sha256, save_json
from yapit.common_utils.mediautils import decode_audio
from yapit.common_utils.modelutils import load_model
from yapit.constants import DEFAULT_SPEECH_MODEL
from yapit.enrollment import load_voice
from yapit.storage import Workspace
from yapit.training.checkpoints import prepare_resume_model, validate_checkpoint


def train(
    workspace: Workspace,
    name: str,
    upstream: Path,
    *,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-6,
    model: str = DEFAULT_SPEECH_MODEL,
    resume: Path | None = None,
) -> dict:
    if model not in (DEFAULT_SPEECH_MODEL, "Qwen/Qwen3-TTS-12Hz-0.6B-Base"):
        raise ValueError("Training model must be the official Qwen3-TTS 1.7B or 0.6B Base.")
    if not 1 <= epochs <= 30 or not 1 <= batch_size <= 16 or not 1e-8 <= learning_rate <= 1e-3:
        raise ValueError("Invalid training parameters.")
    if resume is not None:
        resume = resume.resolve()
        validate_checkpoint(resume, workspace.profile_dir(name).name)
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
    voice = load_voice(workspace.data, name)
    name = voice.name
    if len(voice.samples) < 10:
        raise ValueError(
            "Fine-tuning needs at least 10 curated distinct clips (project guardrail)."
        )
    voice_dir = workspace.voice_audio_dir(name)
    job = workspace.output("training", name)
    transcript_dir = job.transcripts
    checkpoint_dir = workspace.voice_path(name).parent / "checkpoints" / job.name
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
    ordered = sorted(records, key=lambda row: file_sha256(Path(row["audio"])))
    heldout = max(1, len(ordered) // 10)
    splits = {"train": ordered[heldout:], "holdout": ordered[:heldout]}
    for split, rows in splits.items():
        job.path(f"{split}.jsonl", transcript_dir).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    total = sum(len(decode_audio(Path(row["audio"]), max_seconds=20)) / 24000 for row in records)
    if total < 60:
        raise ValueError(
            "Fine-tuning needs at least 60 seconds; 5-15 clean minutes is recommended."
        )
    save_json(job.path("status.json"), {"status": "training", "upstream_commit": revision})
    continuation = None
    try:
        # Upstream saves checkpoints by copying a local model directory, not a Hub identifier.
        local_model = load_model(
            lambda model, **kwargs: snapshot_download(repo_id=model, **kwargs),
            model,
        )
        config = json.loads((Path(local_model) / "config.json").read_text(encoding="utf-8"))
        if config.get("tts_model_type") != "base":
            raise ValueError("Training initialization must be a Base model.")
        initialization = local_model
        if resume is not None:
            continuation = tempfile.TemporaryDirectory(prefix="yapit-resume-", dir=workspace.root)
            initialization = continuation.name
            prepare_resume_model(Path(local_model), resume, Path(initialization), name)
        provenance = {
            "upstream_commit": revision,
            "base_snapshot": local_model,
            "base_model": model,
            "sft_runner_sha256": file_sha256(Path(__file__).with_name("sft_runner.py")),
            "voice": name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "holdout_clips": heldout,
            "training_clips": len(splits["train"]),
            "automatic_quality_approval": False,
            "resume_checkpoint": str(resume) if resume else None,
            "resume_weights_sha256": file_sha256(resume / "model.safetensors") if resume else None,
            "optimizer": "new AdamW; model-weight continuation only" if resume else "new AdamW",
        }
        save_json(job.path("training.json"), provenance)
        commands = [
            [
                sys.executable,
                str(training_dir / "prepare_data.py"),
                "--device",
                "cuda:0",
                "--tokenizer_model_path",
                str(Path(local_model) / "speech_tokenizer"),
                "--input_jsonl",
                str(job.path("train.jsonl", transcript_dir)),
                "--output_jsonl",
                str(job.path("train_codes.jsonl", transcript_dir)),
            ],
            [
                sys.executable,
                "-m",
                "yapit.training.sft_runner",
                str(training_dir / "sft_12hz.py"),
                str(job.path("tensorboard")),
                "--init_model_path",
                initialization,
                "--output_model_path",
                str(checkpoint_dir),
                "--train_jsonl",
                str(job.path("train_codes.jsonl", transcript_dir)),
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
        with job.path("training.log").open("w", encoding="utf-8") as log:
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
            for path in sorted(checkpoint_dir.glob("checkpoint-epoch-*"))
            if (path / "model.safetensors").is_file() and (path / "config.json").is_file()
        ]
        if not checkpoints:
            raise RuntimeError("Training produced no usable checkpoint.")
        result = {
            "status": "trained_needs_listening_review",
            "job": str(job.metadata),
            "name": job.name,
            "status_file": str(job.path("status.json")),
            "checkpoints": checkpoints,
            "transcripts": str(transcript_dir),
            "resumed_from": str(resume) if resume else None,
            "next": "Review held-out audio, then choose an epoch with voice-select-checkpoint.",
        }
        save_json(job.path("status.json"), result)
        return result
    except Exception as exc:
        save_json(job.path("status.json"), {"status": "failed", "error": str(exc)})
        raise
    finally:
        if continuation is not None:
            continuation.cleanup()


def resume_checkpoint(
    workspace: Workspace,
    name: str,
    checkpoint: Path,
    upstream: Path,
    *,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-6,
    model: str = DEFAULT_SPEECH_MODEL,
) -> dict:
    """Continue fine-tuning model weights, not optimizer/epoch state or narration selection."""
    return train(
        workspace,
        name,
        upstream,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model=model,
        resume=checkpoint,
    )
