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

Sources: [Qwen model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base),
[Pocket models](https://github.com/kyutai-labs/pocket-tts),
[Pocket training](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md).
Adapter setup: [Yapper](yapper/README.md#choose-a-model);
other optional backends/history: [research archive](learn_and_research.md).
