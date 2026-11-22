# Training development

[Back to YapIt](../../README.md) | [Usage](README.md) |
[Shared test policy](../dev.readme.md#testing-and-diagnostics)

## Ownership

`trainer.py` owns `train` and weights-only `resume_checkpoint`;
`checkpoints.py` owns validation/selection and Base-compatible initialization.
`dataset.py` owns training-only manifests, legacy migration and short-clip gates;
`session.py` pairs the 16 static prompts with resumable takes.
`desktop.py` owns collection/model/job controls; `jobs.py` submits directly on
Linux or through an explicitly configured WSL Python on Windows.
CLI forwards explicit requests. Yapper consumes selected profile metadata,
not the trainer. SDK imports remain inside requested training calls.
Shared hashing, media, model snapshot and storage primitives stay in their owners.

## Execution and continuation

Platform/CUDA/BF16/FlashAttention and reviewed-checkout gates precede optimization.
Clips are sorted by hash; `max(1, count // 10)` targets are held out.
Upstream preparation/SFT run as bounded subprocesses against a resolved local
Base snapshot and its tokenizer, not a Hub ID used as a copy path.

Owned `sft_runner.py` supplies per-job TensorBoard logging without editing upstream.
For differing text/codec embedding widths, a temporary hook uses the model's own
projection; equal-width models are unchanged. Hooks are removed after execution.
Provenance records upstream/base revision and launcher hash. Failure retains
status/logs; selecting inference weights is always a separate call.

Continuation validates CustomVoice speaker ID 3000, Base architecture and tensor
keys/shapes. It retains trained tensors and restores missing `speaker_encoder.*`
from Base in an owned temporary initialization. Root weight shards/indexes are
excluded from copied assets; nested tokenizer assets stay. Immutable assets may
be hard linked, otherwise copied; config/merged weights are newly written.
Source checkpoints and selected profiles are not modified; temporary initialization
is removed on success/failure.

The selected Base snapshot is resolved per run, not recovered from an external
checkpoint's provenance. Compatible shapes do not prove identical lineage.
Upstream initializes a new optimizer; no scheduler/step-state restoration
or automatic checkpoint promotion occurs.

Training targets come only from the training dataset. Conditioning uses the
profile's separately enrolled reference; legacy profiles may still share that
reference with their historical training set. The holdout is a practical
listening check, not a strict unseen-reference study.
Prepared-code reuse and atomic latest-selected retention are proposed
optimizations, not implemented behavior.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest src\training\tests -q
```

Fast units fake SDKs/tensors and intercept subprocesses for execution gates,
snapshot/holdout behavior, checkpoint compatibility, source immutability and
cleanup. Media/enrolled/process workflows are opt-in integration checks.
Do not run optimization/downloads/GPU work merely to validate refactoring.

The training execution units now construct profiles/datasets directly and mock
media, SDKs and subprocesses. Collection/migration units cover reference
isolation and byte-preserving relocation. WSL submission tests mock path
translation and the job process; no WSL/GPU runtime is probed during validation.
