# Deferred work

[Back to YapIt](../../README.md)

## CosyVoice - deferred, not an installed backend

Complete in order; no new model execution or provisioning is authorized by this list.

1. **Resolve shared-environment compatibility.** Previous setup used Python 3.10,
   Torch/torchaudio 2.3.1, Transformers 4.51.3 and NumPy 1.26.4. These conflict
   with the main Torch 2.9.1/Transformers 4.57.3 stack and Pocket's NumPy >=2.
   Verify a supported combination in the existing CPU/GPU environment; do not
   downgrade it or create a backend-specific environment.
2. **Recheck model/source/license and assets.** Prior experiments used
   `FunAudioLLM/CosyVoice2-0.5B` and `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`,
   a local upstream checkout, exact reference words and local weights. Select
   a verified compatible release; archived experiments are not current support.
3. **Implement through the generic adapter contract.** Reuse `yapit.tts` /
   `ModelBackend`, lazy loading, reference conditioning, explicit device handling,
   close/cleanup and cache identity; emit finite mono float32 24 kHz PCM.
   Add mocked routing, lifecycle and failure tests, not core workflow branches.
4. **Validate only when separately authorized.** Measure cold/warm stage timings,
   memory and held-out word/voice/delivery quality. Historical CPU synthesis
   worked; shared-runtime compatibility and GPU/quality acceptance remain open.

The adapter, isolated bridge, trial script and requirements were removed from the
active tree. Historical findings remain in the append-only
[research archive](../learn_and_research.md); existing local caches are not removed.
