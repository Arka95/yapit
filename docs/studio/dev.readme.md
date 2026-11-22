# Desktop studio development

[Back to YapIt](../../README.md) | [Usage](README.md) |
[Shared development policy](../dev.readme.md#testing-and-diagnostics)

The existing enrollment launcher/entry point is preserved. `EnrollmentApp`
owns the Tk root, notebook, shared status and queued background-operation
dispatcher; its recording controls remain in the first tab. `StudioTabs`
adds Audio/Reel views and snapshots widget inputs before starting workers.
Blocking generation, decoding, recognition and rendering stay off Tk's thread.
One operation runs at a time; close waits for completion. Enrollment and Training
share recording controls with separate sessions and microphones; leaving either
releases its prepared microphone. Tab changes during recording return to its owner.

| Owner | Responsibility |
|---|---|
| `studio/desktop.py` | Tab controls, reviewed script state, transport, waveform/cut display and placeholder preview |
| `studio/workflow.py` | SDK-free adapter discovery; owned temporary playback audio; explicit timeline analysis and video rendering |
| `studio/player.py` | Lazy Windows MCI WAV playback, seeking and handle cleanup |
| `yapper/model_catalog.py` | Local model/config metadata inventory and voice-compatible checkpoint choices |

Audio generation calls `yapper.generate_audio` with the selected adapter/model,
without model-specific UI branches. An unchanged AI `Script` retains visual
beats; edits are parsed through `Script.from_text`. Generated results hand off
their audio and timeline paths. Imports initially decode only, with the
180-second reel-input limit; explicit analysis delegates to offline Rescript.
Both flows render through Editor's existing `generate_video` facility, preserving
audio and avoiding a second TTS pass. Preview frames use Editor's placeholder
renderer at thumbnail resolution.

Each studio session owns a unique temporary directory inside the configured
workspace. The player closes before media replacement/cleanup. Durable sources
and generated results are never removed. Configured workspace/data roots are
shared with enrollment. A read-only dropdown maps friendly labels to discovered
model identifiers; widget inputs never become arbitrary paths. Discovery checks
the configured models directory, its hub cache, configured Hugging Face caches,
and the selected profile's checkpoint folders. Qwen choices require compatible
config metadata and main/tokenizer weight files, including all indexed shards.
CustomVoice choices must contain the selected speaker. Pocket language choices
require cached clone-capable weights, excluding the non-cloning repository.
SDKs are not imported, and no catalog or model assets are fetched.
Backend/profile changes discard stale choices. The selected model is explicitly
forwarded, so it cannot inherit another backend's configured model identifier.
No new dependency, runtime, CLI command or MCP tool is required.

Run `.\.venv\Scripts\python.exe -m pytest src\studio\tests src\enrollment\tests\test_units.py src\yapper\tests\test_model_catalog.py -q`
for focused rapid units: adapter forwarding, reviewed-text preservation,
generated timing reuse, explicit offline analysis, replacement/failure cleanup,
local model filtering, stale model-choice replacement, waveform transients and
mocked native playback. No model, media subprocess,
GUI event loop, microphone or real render runs in these tests. Actual Windows
playback/layout and model quality still require explicit manual validation.
