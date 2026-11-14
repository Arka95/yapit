# Licensing and commercial use

[Back to YapIt](../README.md)

Verified November 14, 2026. This is a practical licensing guide, not legal advice
or a complete dependency/license audit.

## YapIt code

YapIt's original source code and associated project documentation are licensed
under [MIT](../LICENSE), matching the package metadata. Commercial use,
modification and redistribution are permitted when the license and copyright
notice are retained. MIT does not reserve commercial exclusivity: other people
can also reuse and sell the MIT-covered code.

Recordings, voice/profile data, trained weights and generated media are not
licensed by this software license unless separately stated. A demo profile
name is not permission to clone its speaker. Customers should enroll their own
recordings rather than treat the repository's `testuser` profile as their voice.

## Models and dependencies

YapIt's license does not replace third-party licenses or access conditions.
Model assets are downloaded separately, not bundled in the source package.

| Component | Upstream license / conditions | Distribution guidance |
|---|---|---|
| Default Qwen3-TTS 1.7B Base weights | Apache-2.0 on the official model card | Preserve applicable license/notices and identify modifications when redistributing covered material |
| Pocket TTS SDK | MIT in the upstream repository | Retain its license/copyright notices when redistributing the SDK |
| Pocket TTS weights | CC-BY-4.0 on the official model card, with additional access/use conditions shown by the provider | Review the exact selected model/revision; credit Kyutai, link the license and describe modifications when sharing licensed material; access may require accepting the provider's conditions |
| Other adapters, tokenizers, dependencies and external tools | Their own licenses | Review the versions actually shipped, especially before redistributing binaries, checkpoints or a bundled installer |

This table does not mean every model variant shares one license, that every
dependency is MIT, or that a model's weight license automatically determines
the copyright/license of every generated audio/video output. Input recordings,
scripts, footage and music retain their own rights and conditions.


## Marketplace skills

The local YapIt runtime and any marketplace skills are different deliverables.
No marketplace skill package is currently included or published by this repository.

For a future paid skill, keep the separately authored skill instructions and
support files under an explicit, separate license rather than publishing them
under this repository's MIT grant if restricted redistribution is intended.
Marketplace terms vary; review the actual platform's distribution, buyer-license
and refund terms before publishing. Do not assume one marketplace's terms apply
to another.

These platform terms do not revoke rights already granted over MIT-covered
code or other openly licensed material. Preserve upstream notices in any reused
material. Do not copy the marketplace's platform terms into a custom EULA or invent
different refund rules: use the actual terms at publication time.

## Official references

- [MIT license guidance](https://choosealicense.com/licenses/mit/)
- [Qwen3-TTS 1.7B Base model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0)
- [Pocket TTS SDK license](https://github.com/kyutai-labs/pocket-tts/blob/main/LICENSE)
- [Pocket TTS model card and access conditions](https://huggingface.co/kyutai/pocket-tts)
- [CC-BY-4.0 summary and legal-code link](https://creativecommons.org/licenses/by/4.0/)
