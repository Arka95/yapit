# Training

The training module owns explicitly requested personal-voice fine-tuning and
manual checkpoint selection. It is separate from [Yapper](../yapper/README.md)
inference and [enrollment](../enrollment/README.md). Enrollment/reference cloning
does not train model weights. See the [developer guide](dev.readme.md) and
[project index](../../README.md).

## Requirements and commands

Actual fine-tuning requires Linux or WSL2, an NVIDIA CUDA GPU with bfloat16
support, existing `.[train]` dependencies, compatible FlashAttention and a clean,
reviewed official Qwen3-TTS checkout containing its finetuning scripts.
Run in that environment with its own valid paths:

```text
yapit voice-train arka --upstream <reviewed-Qwen3-TTS-directory> --epochs 3 --batch-size 1
yapit voice-train arka --upstream <reviewed-Qwen3-TTS-directory> --model Qwen/Qwen3-TTS-12Hz-0.6B-Base
yapit voice-train arka --upstream <reviewed-Qwen3-TTS-directory> --resume <validated-checkpoint-directory>
yapit voice-select-checkpoint arka --checkpoint <reviewed-checkpoint-directory>
```

The Python APIs are `train`, `resume_checkpoint` and `select_checkpoint` from
`yapit.training`. Old `use_checkpoint` and hidden `voice-use-checkpoint` remain
inference-selection compatibility aliases, not training resumption.
Importing the module or generating narration does not
start a training job. Native Windows training fails before loading/downloading
training dependencies or models.

`--model` explicitly selects the official 1.7B or 0.6B Base Hub model;
the default remains `Qwen/Qwen3-TTS-12Hz-1.7B-Base`. Python `train` and
`resume_checkpoint` accept the same `model` keyword. It is independent of
the inference setting `YAPIT_MODEL`. Smaller weights reduce memory demand,
but do not guarantee training fits: optimizer states and activations also need
GPU memory. The selected model ID and resolved snapshot are recorded in provenance.

Project guardrails require at least 10 curated clips and 60 seconds of recordings.
Five to fifteen clean minutes is a recommendation, not a quality guarantee.
The wrapper sorts clips by file hash, holds out `max(1, clip_count // 10)` clips,
prepares codec tokens and runs upstream supervised fine-tuning.
Defaults remain three epochs, batch size one and learning rate `2e-6`.
Training does not fall back to CPU when CUDA is unavailable. The CPU still
prepares data and loads weights; automatic CPU fallback applies to normal
setup/narration, not this fine-tuning operation.

## Continue from checkpoint weights

`voice-train --resume DIRECTORY` continues model weights with a **new optimizer**.
It does not restore optimizer/scheduler state, the old epoch counter or an
interrupted run's exact trajectory. `--epochs` is the new run's epoch count;
the reviewed upstream checkout remains required.

The checkpoint must contain the profile's named CustomVoice speaker at upstream
speaker ID 3000, the required config/model files, matching Base architecture,
and compatible tensor keys/shapes. The wrapper retains fine-tuned tensors and
restores missing Base speaker-encoder weights/config in a temporary initialization
directory. Source checkpoint/Base files and the selected profile are not modified;
temporary initialization is removed on success or failure.

Continuation uses the explicitly selected Base snapshot (1.7B by default),
not `YAPIT_MODEL`. Supply the same `--model` used for the original training run;
existing architecture/tensor checks reject incompatible checkpoint sizes.
Architecture/shape checks do not establish that an external checkpoint used
the identical original Base revision; use checkpoints from the same Base lineage
and review the resulting voice. Merging weights requires additional CPU RAM and
disk, potentially model-sized copies when hard linking is unavailable.
These compatibility checks do not guarantee training or voice quality on a
particular GPU.

## Checkpoints and review

Training does not automatically replace the profile's current reference/checkpoint.
Successful runs return `trained_needs_listening_review`. Compare held-out phrases
and select a checkpoint explicitly; selection verifies CustomVoice model type,
the requested speaker and required model/config files.
Yapper then uses that checkpoint for custom-speaker inference.

JSONL datasets go to `data\transcripts`; training status/logs and base/upstream
provenance go to `workspace\outputs`; TensorBoard uses a per-job
`<job-name>.tensorboard` directory there. Checkpoints go to
`data\<profile>\checkpoints\<job-name>`. Existing data/archive exclusions remain
unchanged; training does not stage or upload recordings.

The wrapper can download the base Qwen snapshot when needed. This is separate
from hosted script-writing APIs; training does not invoke a script LLM.
Importing the package does not run training or provision a GPU environment.
File validation and a successful training run do not establish improved voice
quality; listening review remains required.
