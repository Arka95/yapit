# Training

[Back to YapIt](../../README.md) | [Developer guide](dev.readme.md)

Explicit Qwen voice fine-tuning and manual checkpoint selection.
[Enrollment](../enrollment/README.md) and reference cloning do not train weights.

## Requirements and commands

Use **Linux/WSL2**, CUDA with bfloat16 support, compatible FlashAttention,
`.[train]` dependencies and a clean, reviewed upstream Qwen3-TTS checkout.
Native Windows training is rejected. Run inside that environment with valid paths:

```text
yapit voice-train myvoice --upstream <reviewed-Qwen3-TTS-directory> --epochs 3 --batch-size 1
yapit voice-train myvoice --upstream <reviewed-Qwen3-TTS-directory> --model Qwen/Qwen3-TTS-12Hz-0.6B-Base
yapit voice-train myvoice --upstream <reviewed-Qwen3-TTS-directory> --resume <validated-checkpoint-directory>
yapit voice-select-checkpoint myvoice --checkpoint <reviewed-checkpoint-directory>
```

At least 10 curated clips/60 seconds are required; 5-15 clean minutes is a
recommendation, not a quality guarantee. Defaults are three epochs, batch one
and learning rate `2e-6`, with about 10% held out. Training never falls back
to CPU and may download a missing Base snapshot.

`--model` selects official 1.7B (default) or 0.6B Base training weights,
independently of inference's `YAPIT_MODEL`. Batch one does not guarantee fitting
a particular GPU.

## Continuation and review

`--resume` continues compatible checkpoint **weights with a new optimizer**,
not the old optimizer/scheduler/epoch state. Supply the same Base model lineage;
architecture/tensor checks reject incompatible sizes but cannot prove lineage.
Temporary restoration can need additional model-sized RAM/disk.

Training returns `trained_needs_listening_review` and never automatically changes
the profile. Compare held-out wording/identity/pacing, then explicitly select.
File validation alone is not voice-quality approval.

Datasets go to `data\transcripts`, checkpoints to
`data\<profile>\checkpoints\<job-name>`, and logs/provenance/TensorBoard to
`workspace\outputs`. Public Python APIs are `train`, `resume_checkpoint` and
`select_checkpoint`; see the developer guide for compatibility/cleanup details.
