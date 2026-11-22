# Audio model shortlist

[Back to YapIt](../README.md) | [Detailed research](learn_and_research.md)

English personal-voice narration. Sizes measured from local weight files on
November 14, 2026; they are disk sizes, **not RAM/VRAM requirements**.

| Model used in YapIt | Parameters (nominal) | Local weight size | Trainable in YapIt? | Salient features |
|---|---|---|---|---|
| Qwen3-TTS 0.6B Base / selected `testuser` epoch-2 | 0.6B | 2.49 GB including speech tokenizer | Yes | Existing trained baseline; reference cloning or selected custom-voice checkpoint; CUDA acceleration. Base delivery follows the reference, not independent emotion instructions. |
| Pocket `english_2026-10` | About 100M | 219 MB main weight file | No | CPU-first reference cloning; used for the README demo and preferred by the owner for that clip. Reusable voice conditioning; no Qwen checkpoint reuse. |

Training means the repository's supported workflow: Qwen full-weight training
on Linux/WSL2. Pocket has upstream training recipes, but no YapIt trainer.
Sizes exclude SDKs/caches; Qwen includes an additional tokenizer weight file.
One listening preference is not a controlled quality comparison.

## Reference-conditioning decision - November 22, 2026

| Current position | Implementation / acceptance criterion |
|---|---|
| Enroll with one fixed, phonetically varied paragraph, approximately one minute. | One 45-90s recording plus its exact transcript becomes the profile's only reference. No silence/duration ranking, merging or reference audition is performed. |
| Pocket and Qwen Base use one reference; the selected Qwen CustomVoice checkpoint does not use a reference prompt at inference. | Pocket prepares one audio/state prompt; Qwen Base uses matching reference text/audio in ICL mode. Qwen list inputs are separate batch entries. |
| Keep the 16 short passages in the Training module and tab. | Training targets/manifests are separate from the enrollment reference. Existing datasets/checkpoints are preserved; Train and checkpoint selection require explicit actions. |
| Predictable onboarding is a product requirement; likeness and speed remain unmeasured. | The owner records once and confirms the transcript. No local inference, reference comparison or training was run for this change. |

See the [production enrollment decision](learn_and_research.md#production-enrollment-decision---november-22-2026)
for the superseded proposals, source-backed research and migration details.

## Implemented

| Capability | Canonical guide |
|---|---|
| Qwen default, explicit Pocket selection and trusted plugins | [Yapper usage](yapper/README.md#choose-a-model) |
| Lazy lifecycle/identity contract, one-model worker and bounded cache | [Yapper development](yapper/dev.readme.md#adapter-contract) |
| Shared runtime and adapter dependency-compatibility policy | [Model onboarding](yapper/dev.readme.md#model-onboarding) |
| Qwen full-weight training, weights-only continuation and manual selection | [Training](training/README.md) |

Adapters do not make weights interchangeable. Reference conditioning is not
fine-tuning; smaller weights do not establish better identity or lower measured
latency. New backends must fit the shared runtime before being enabled.


Sources: [Qwen model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base),
[Pocket models](https://github.com/kyutai-labs/pocket-tts),
[Pocket training](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md).
Adapter setup: [Yapper](yapper/README.md#choose-a-model);
other optional backends/history: [research archive](learn_and_research.md).
