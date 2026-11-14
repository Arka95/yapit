---
applyTo: "**"
---

# Model decisions and research

- Maintain `docs\analysis.md` as a small table of two or three suggested models
  used in this repository: parameter count, weight size, whether training is
  supported in YapIt, and brief salient features. Distinguish disk size from
  runtime memory and upstream training support from repository integration.
- After model research or a selection decision, append dated findings and
  decisions, then refine the guide into the current position without repetition.
  Separate actual settings from suggested experiments; never claim a quality or
  performance winner without local evidence.
- Keep detailed research, equations, sources and historical decisions in
  `docs\learn_and_research.md`. It is append-only and excluded from README/docs
  compaction: never rewrite, shorten, reorder or remove its existing content.
  Append dated corrections instead of editing historical claims.
- Before compacting `analysis.md`, append its research/equations to the archive
  and verify that existing archive content and transferred material are preserved.
- Verify latest appropriate stable 2026+ model releases from official sources.
  Documentation work does not authorize inference, downloads, weight conversion,
  training, checkpoint replacement or dependency installation.
- Onboard TTS SDK families through the existing managed backend registry/plugin
  contract in the selected CPU/GPU environment. Defer incompatible SDKs to
  `docs\todo`; do not create model-specific interpreters/runtimes.
  Do not add model-specific branches to Pipeline, Editor or core CLI workflows.
  Model IDs select compatible variants; new architectures need trusted adapters.
