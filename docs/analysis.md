# Model analysis and configuration

Updated: **November 8, 2026**. Scope: **English, Arka's enrolled voice, engaging
travel/product reels**, RTX 4070 SUPER or CPU travel use. Decisions below are
current; detailed research, equations and superseded findings live in the
append-only [research archive](learn_and_research.md).

## Model choices and decisions

| Model | Decision | Main constraint |
|---|---|---|
| Qwen3-TTS 0.6B Base + Arka epoch-2 | Keep the existing trained baseline; managed inference and CUDA-graph acceleration are implemented. | Base clones reference delivery, not independent emotion instructions. Existing training is full-weight, not LoRA. |
| Pocket `english_2026-10`, about 100M | Owner prefers the Puerto Princesa generation over prior Qwen output; keep it as the preferred candidate for similar narration, with Qwen reversible. CPU/reference generation works with SDK v4.0.1. | Single-clip subjective feedback, not a matched quality benchmark or complete word/identity audit. Cloning weights require publisher access approval and local Hugging Face login. |
| Pocket `english_18m_2026-10`, 18.2M including decoder | First smaller-model comparison; retain the standard model as quality fallback. | Do not promote until Arka accepts identity, words and delivery. |
| Pocket `english_7m_2026-10` / `english_2m_2026-10`, 6.9M / 1.9M | Later size experiments only if 18M is still too costly. | Smaller does not prove better latency or adequate narration quality. |
| VoxCPM2, 2B, April 2026 | First future challenger when independent style instructions plus reference identity are required. | Larger runtime; adapter, Windows behavior and Arka quality remain unverified. |
| IndexTTS 2.5, August 2026 | Specialist option for separate emotion references/vectors and duration controls. | Custom-license review, isolated runtime and adapter required. |
| CosyVoice2-0.5B | Explicit owner exception to the release-date rule; integrated through the generic registry/isolated-runtime bridge. Arka-reference CPU synthesis works. | December 2024 model, not a 2026 release. Current CPU setup is costly; quality needs owner listening review. |
| CosyVoice3 `2512` | Same SDK-family adapter can select the cached model; no Pipeline/Editor changes. | December 2025 release; selection is supported but real YapIt synthesis has not been validated. |
| Retired Chatterbox; older research models | Historical research, not current adoption choices. | Not current agent-selected candidates. |

**Selection policy:** agent-selected models must be the verified latest
appropriate stable **2026+ releases**, checked against official code/model cards.
Recent repository edits are not new weight releases. User-supplied adapters/IDs
are not automatically date-validated. Preserve the accepted checkpoint rather
than silently upgrading or substituting a different speaker.
CosyVoice2 was specifically requested despite its older release. No official
2026 CosyVoice weight release was found; newer SDK/model-card edits do not qualify.

Qwen's recorded Base revision matches current official Hub metadata:
`5d83992436eae1d760afd27aff78a71d676296fc` (last modified January 29, 2026).
That confirms the source revision, not that Arka's adaptation is quality-optimal.

## Configuration and parameters to focus on

| Control | Current setting / recommended starting point | Why it matters |
|---|---|---|
| Profile and language | `arka`; English. Keep exact reference transcript and approved script. | Identity, accent and names come before speed. |
| GPU baseline | `--backend qwen`, no model override: use selected epoch-2. `YAPIT_DEVICE=cuda:0`. | Preserves the trained voice; explicit GPU selection avoids automatic CPU fallback. |
| Qwen precision / attention | Adapter chooses BF16 where supported, otherwise FP16; SDPA; CUDA graphs enabled by default. CPU uses FP32. | Existing implementation, not a new precision experiment. |
| Qwen generation | `language="English"`, `max_new_tokens=2048`; retain SDK sampling defaults. | Token budget is a guard, not a duration target. Temperature/top-p are not emotion controls. |
| Pocket baseline | `--backend pocket --model english_2026-10`; CPU; reuse reference conditioning or exported voice state. | Avoids fine-tuning and repeated conditioning in a resident adapter. |
| Pocket size experiment | Change only `--model` to `english_18m_2026-10` first. | Isolates model-size effects from changed references and scripts. |
| CosyVoice comparison | `--backend cosyvoice --model FunAudioLLM/CosyVoice2-0.5B`; isolated Python 3.10, Torch 2.3.1+cpu, actual FP32 components; speed 1.0. | Uses Arka's reference and exact transcript, not the Qwen checkpoint. Optional normalization/downloads and TRT/vLLM are excluded. GPU has not been validated. |
| KV caching | Already used in Qwen's autoregressive decoder/CUDA-graph loop, Pocket's streaming transformer and CosyVoice's LLM decoder. | Reuses prior attention keys/values within generation; not the same as keeping weights loaded or caching finished speech. Pocket also preserves reusable reference-prefix state. |
| vLLM | Not enabled. Consider a supported isolated CosyVoice GPU experiment only after a plain-GPU baseline and stage timings. | Accelerates its LLM, not the complete flow/vocoder pipeline. Single-request gains need measurement; batching is not an automatic win for sequential reels. Official GPU runtime requires Linux/WSL and compatible dependencies. |
| Reference / delivery | Clean consented Arka recording with verified words; real expressive reference where supported. | Profile labels do not prove emotion control; do not invent an emotion flag for Base. |
| Duration / output | Natural delivery with explicit pauses; preserve existing word/duration gates; mono float32 24 kHz adapter output. | Never omit words or over-stretch to force a video fit. |
| Residency / cache | Opt-in worker only; otherwise reuse conditioning across beats and close the model/runtime after generation. | Old Qwen worker was stopped. No resident worker remains; custom configs need immutable weight revisions. |
| Training baseline, not tuned optimum | 15 training clips + 1 holdout; 3 epochs; batch 1; learning rate `2e-6`; new AdamW on continuation. | Actual Arka run. Reserve holdout for listening/text checks; do not confuse weight continuation with optimizer resume. |
| Future Pocket adaptation | Official same-language teacher fine-tuning, then optional six-layer distillation; Linux/WSL2. | Not implemented in YapIt's trainer. Upstream `finetune.yaml` uses LR `2e-5`, Muon LR `5e-4`, batch 64 and 20k steps: template settings, not suitable-by-default for 15 clips or 12 GB VRAM. |

Model selection uses `audiogen` / `voice-worker` flags, or
`YAPIT_TTS_BACKEND` / `YAPIT_TTS_MODEL`. Other workflows can inherit worker
startup selection. Different architectures require built-in or installed
`yapit.tts` adapters, not merely a different model ID. Training remains
Qwen-specific. Restart a legacy worker before requesting explicit selectors.
Incompatible SDKs use the reusable private JSON/WAV runtime bridge; new
architectures still need their own trusted adapter, not just a model ID.
CosyVoice2's inspected LLM input orders reference transcript, new script, then
reference speech tokens. Changing the script therefore prevents whole-reference
speech KV-prefix reuse; cached extracted conditioning is a separate optimization.

## English-only, one-profile size and weight optimization

**Current owner preference:** standard dated English Pocket with Arka reference
conditioning, based on the Puerto Princesa clip. Keep Qwen 0.6B epoch-2 as the
reversible GPU baseline, with its implemented precision/acceleration and one
resident model. **Smaller experiment:** reusable Arka voice state plus Pocket 18M;
it has not been generated or judged. Global defaults have not been changed.

| Optimization | Decision and limit |
|---|---|
| Select a smaller English-native model | Prefer released Pocket English variants over trying to carve English out of multilingual Qwen tensors. This actually selects a smaller architecture. |
| Persist model-specific profile conditioning | First profile-reuse improvement: prepare Arka once per model revision/reference/style. Pocket already accepts an exported voice-state file; automatic persistent conditioning caches for all adapters are proposed. |
| Keep shared profile assets, not shared model tensors | Reuse consented recordings/transcripts and preprocessing where compatible. Speaker embeddings, acoustic tokens and KV state are model-specific, including across incompatible variants; a voice state is not full-weight fine-tuning. |
| Reuse exact finished speech | Worker caching already keys model/voice identity plus exact text. Extending this to one-shot generation is proposed; preserve backend/revision/reference/settings in the key. |
| Bounded model residency | An explicitly owned adapter can serve multiple jobs, then close; a worker is opt-in. Do not keep all models loaded merely because the profile is shared. Automatic idle unloading is not implemented. |
| Remove duplicate exports, not essential layers | Only Arka epoch-2 is retained locally, about 2.33 GiB. Future atomic latest-selected retention and shared immutable codec assets are proposed, not implemented. Preserve the selected checkpoint on failure. |
| Quantize after the baseline | Pocket v4.0.1 exposes dynamic INT8 transformer quantization upstream; YapIt's adapter does not expose it yet. Qwen INT8/INT4 is not implemented or validated here. Verify operator support, graph compatibility, identity and word fidelity before adoption. |
| LoRA for future adaptation | A small per-profile update can reduce training/export cost, but does not shrink the resident base. Qwen's current trainer is not LoRA; VoxCPM documents a separate recipe, untested here. |
| Language/speaker pruning | Selecting English or one speaker does not shrink dense weights. Do not delete tokenizer rows, codec codebooks or layers; safe structural pruning needs architecture changes and recovery/evaluation. |
| Distill only if released small models fail | A smaller student trained to imitate an accepted teacher is a research project, not a shortcut from 15 clips. Cover place/brand names, numbers, accent and restrained expressive delivery in held-out tests. |

**Tune only after a demonstrated failure:** model size/precision for memory;
reference quality for identity; supported style controls for delivery; chunk
length and pauses for continuity/timing. Do not simultaneously change all four.
Training proposals are reusable codec preparation, measured stage timings,
atomic publication and latest-selected retention; no new training is authorized.
Profile caching cannot remove neural generation cost: CosyVoice2 took 138.58s
overall with 26.36s load/reference preparation. Even removing that entire stage
would leave 112.22s; persisting conditioning alone saves less. A separately
authorized GPU/stage benchmark is the next speed investigation, not a claimed gain.
Pocket's published recipe fine-tunes a 24-layer English teacher before
distillation; the current teacher shortlist includes `english_2026-09_24l`.
Its documented consumer training setup uses about 16 GB VRAM, so fitting this
12 GB card requires a separately verified memory/configuration plan. The 200-hour
quickstart is from-scratch training, not a stated personal fine-tuning minimum.

## Evidence and acceptance

- Controlled Qwen CUDA-graph benchmark: **33.65s -> 10.10s**, identical waveform
  hash; one workload, not a general speed guarantee. No new model benchmark ran.
- Managed-backend checks: **16 mocked units passed in 0.60s**; lint/format checks
  passed. These prove routing/lifecycle behavior, not Pocket voice quality.
- First Pocket CPU run, four threads: **7.13s load**, **16.99s synthesis** across
  six beats, **31.02s pipeline** after weight download. Output: **49.55s narration /
  50.30s guide**, 1080x1920 at 30 fps; video decodes and narration is finite,
  non-silent and unclipped. One workload, not a matched Qwen comparison.
- Owner listening feedback on November 8: Pocket's Puerto Princesa generation
  is better than previous Qwen output. This supports preference for this clip,
  not universal superiority or a completed pronunciation/identity evaluation.
- CosyVoice2: **8.52s** natural-speed Arka-reference sample, **138.58s pipeline**
  including **26.36s load** on CPU. WAV/MP3 are finite, non-silent and unclipped;
  the child exited and no worker state remains. **35 mocked units passed in 1.10s**,
  with lint/format checks passing. This validates integration, not voice quality,
  GPU readiness or a matched performance comparison.
- Keep an output only when Arka accepts the voice and natural delivery, every
  approved word/name is preserved, and existing signal/duration gates pass.
  Record warm/cold time and memory separately; smaller weight count alone cannot win.

Local evidence: `src\yapper\backends.py`, `speech.py`, `pocket.py`, `worker.py`;
`src\yapper\cosyvoice.py`, `cosyvoice_runtime.py`, `isolated.py`, `isolated_runtime.py`;
`src\training\trainer.py`, `sft_runner.py`, `checkpoints.py`;
`workspace\outputs\arka.training.training.json`;
session `files\4070-cuda-graph-benchmark.json`.
Pocket run: `workspace\outputs\arka.puerto_princesa.pocket_2026_10.3.status.json`,
its provenance, `.repo-stats\yapper\functions.jsonl`, and session
`files\puerto-princesa-pocket-verification.json`.
CosyVoice run: `workspace\outputs\arka.cosyvoice2.validation.2.status.json`,
its provenance, and session `files\cosyvoice2-validation-verification.json`.
Official revisions, equations and research limitations are in the dated archive;
[training/backend assessment](tts-training-and-backends.md) retains implementation detail.
