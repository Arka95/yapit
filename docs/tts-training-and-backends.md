# Backend and training status

[Back to YapIt](../README.md)

This concise index replaces the old assessment's duplicated implementation,
benchmark and rollout sections. Current choices/measurements remain in
[model analysis](analysis.md); historical evidence remains in the
[research archive](learn_and_research.md).

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

## Proposed, not implemented

Reuse prepared codec data only when audio/text/reference/tokenizer/preprocessing
identity matches. Measure preparation, optimization and export separately.
Publish/validate a new checkpoint before applying a future retention policy;
never remove the selected checkpoint on an incomplete or failed save.
These are separate from changing training recipes or model selection.

Unit success and historical single samples are not a controlled quality
comparison. Require separately authorized held-out listening/wording checks
before claiming a better voice or training configuration.
