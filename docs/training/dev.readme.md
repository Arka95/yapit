# Training development

Source is `src\training`; public imports remain `yapit.training`. See
[usage](README.md) and the [folder-level project layout](../dev.readme.md#flat-source-layout).

## Module boundaries

`trainer.py` contains `train` and weights-only `resume_checkpoint`.
`checkpoints.py` owns checkpoint validation, inference `select_checkpoint` and
Base-compatible initialization. `__init__.py` exports the three public APIs;
`use_checkpoint` remains an inference-selection compatibility alias.
The former loose `src\training.py` is removed. Root CLI only forwards explicit
training/checkpoint requests; Yapper consumes selected profile metadata but does
not import or execute the trainer.

Shared contracts/storage/profile operations remain in `models.py`, `storage.py`
and enrollment/voice adapters. The trainer reuses `common_utils.fileutils` for
streaming hashes and atomic JSON, `mediautils` for decoding/duration validation,
and `modelutils` for the native Hub snapshot factory.

## Preserved execution policy

Platform/CUDA/bfloat16/FlashAttention and reviewed-checkout checks precede
optimization. Training SDKs are imported inside `train`, not at module import.
Clip count/duration guardrails, deterministic holdout, upstream/base revision
provenance, bounds on epochs/batch size/learning rate and subprocess timeouts
remain unchanged.

Upstream `prepare_data.py` and `sft_12hz.py` run as explicit subprocesses.
The owned `sft_runner.py` launcher supplies a per-job TensorBoard logging directory
missing from upstream's Accelerator constructor; it preserves all upstream
training arguments and leaves the reviewed checkout unchanged.
Provenance includes the launcher's SHA-256; data preparation uses the selected
Base snapshot's local speech tokenizer rather than resolving a separate revision.
For smaller Base architectures whose text/codec embedding widths differ,
the launcher applies the model's own text projection through a temporary
embedding hook, matching inference's dimensional conversion. Equal-width
1.7B training is unchanged; hooks are removed on success or failure.
The wrapper passes a local base snapshot directory, never a Hub identifier as
the checkpoint-copy source. Training status/logs persist on failure; profile
promotion requires a separate reviewed `select_checkpoint` call.

For continuation, lazy safetensors reads validate the checkpoint speaker/architecture
and tensor keys/shapes against the resolved Base snapshot. Missing
`speaker_encoder.*` tensors removed by upstream SFT are restored from Base;
trained tensors are retained. Root weight shards/indexes are excluded from copied
assets so loading uses the merged `model.safetensors`, while nested tokenizer
assets are retained. Assets are hard linked from resolved source files where
possible, otherwise copied; config/merged weights are newly written.
The owned temporary Base initialization is removed on both success and failure.
Training provenance records the resume path/weight hash and new optimizer
semantics. The unchanged upstream trainer initializes a fresh optimizer;
there is no optimizer/scheduler/step-state restoration or automatic promotion.
The explicit `model`/CLI `--model` selects the official 1.7B or 0.6B Base,
with 1.7B unchanged as the default. The resolved config must be Base; model ID
and snapshot path are recorded in provenance. Resume uses the same selection
and existing architecture/tensor checks, not the inference `YAPIT_MODEL`.
The Base snapshot is resolved for each run, not recovered from an external
checkpoint's provenance; matching architecture/shapes is not proof of identical
Base weights. Use the same Base lineage and validate quality separately.

A holdout is a practical listening check, not a strict unseen-reference research
evaluation: all rows share the selected reference, which may belong to the
enrollment set. File/type/speaker validation is not automatic audio-quality
approval.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q src\training\tests
```

All training-only tests, including checkpoint selection and unsupported-host
rejection formerly in root voice tests, now live in this module. Its
package-scoped fixtures create synthetic accepted recordings under isolated
temporary roots.

Fake Torch/FlashAttention/Hub SDKs and intercepted upstream subprocesses verify
success/failure status, local snapshot arguments, disjoint train/holdout targets
and no automatic promotion. Tiny fake safetensors cover sharded Base merging,
encoder restoration, retained trained weights/tokenizer assets, rejected
speaker/architecture/key/shape mismatches, source immutability and temporary
cleanup on success/failure. An independent subprocess checks that importing
Yapper does not import training and importing training initializes no model SDKs
or cache. Tests do not run real optimization, downloads or GPU workloads.
