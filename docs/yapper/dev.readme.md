# Yapper development

[Back to YapIt](../../README.md) | [Usage](README.md) |
[Shared test policy](../dev.readme.md#testing-and-diagnostics)

## Ownership and APIs

| Component | Responsibility |
|---|---|
| `script_planner.py`, packaged `script_prompt.txt` | Script parsing, deterministic prompts, explicit cloud writing |
| `backends.py`, adapter modules | Lazy factories, lifecycle/identity contract and validated PCM |
| `worker.py`, `inference.py` | One-model worker/cache and Qwen codebook CUDA graphs |
| `generation.py`, `timeline.py` | Synthesis/publication and measured cuts/approximate captions |

`generate_audio(Script | str, *, workspace, voice_name, seconds, style, ...)`
is the audio-only API. It supports injected `SpeechBackend` or managed
`backend_name`/`model`, not both. `render_audio` takes validated Script/Voice
and reserved outputs, returning `(result, Timeline)`; Pipeline uses it to share
one reel job. CLI narration and MCP delegate here, not to Editor.
The optional module CLI is `python -m yapit.yapper generate`.

Contracts/storage/profile selection remain in their shared owners.
Use `common_utils` for file/media/model primitives, not duplicate implementations.

## Adapter contract

Injected backends need `synthesize(text) -> np.ndarray`. Managed `ModelBackend`
adapters implement `load`, `close`, `synthesize`, `cache_identity` and `describe`.
Constructors stay lightweight; SDK imports/loading occur on demand.
Return finite, non-silent mono float32 24 kHz PCM and enforce bounded chunks.
Callers own injected adapters; one-shot rendering closes adapters it creates.

Factories accept `(voice, voice_dir, *, model=None)`. Register locally with
`register_backend("name", factory)`, or install a trusted plugin:

```toml
[project.entry-points."yapit.tts"]
my-model = "my_tts_adapter:MySpeech"
```

`cache_identity` covers backend/model/version/conditioning and relevant asset
stamps; writing metadata is excluded. `describe` reports actual runtime metadata.
If language, seed, sampling or instruction controls are added, include effective
request settings in waveform-cache identity without needlessly reloading weights.
Qwen-specific training/checkpoint checks stay in Training.

## Model onboarding

Verify model/revision/license/sample rate and reference requirements before adding
an adapter. Keep variants in configuration and implement new architectures through
the shared contract, without model-specific Pipeline/Editor branches.
Adapters use the selected CPU/GPU environment, not a model-specific interpreter.
Resolve SDK compatibility before enabling a backend; defer incompatible SDKs
instead of replacing the working stack or creating an environment per model.
Cover selection, reference reuse, conversion, failure/path handling and cleanup
with mocks; real provisioning/quality comparisons require authorization.

Pocket optionally uses profile `pocket_voice_state`; otherwise the reference
recording. Clone-weight access may require publisher approval and local sign-in.
Custom YAML cache identity does not stamp every referenced weight: use immutable
assets and restart after changing them.

Model comparisons/history belong in [analysis](../analysis.md) and
[research](../learn_and_research.md), not duplicated setup reports.

## Script planning

The public Python contract, exported by `yapit.yapper`, is
`generate_script(prompt: str, seconds: float = 30) -> Script`.
CLI `generate-script` and MCP `generate_script` delegate to it and return JSON.
No image, frame or media input is accepted by this contract.

`load_script` parses approved text/JSON; `build_prompt` prepares writing instructions.
The internal text writer retains voice-rate, duration-buffer and closing-hold
planning for the full reel pipeline. Supplied scripts never call the writer;
Editor owns video templates and rendering.

Explicit script generation or `generate-reel --prompt` needs `.[cloud]`,
`YAPIT_ALLOW_CLOUD=true` and
`OPENAI_API_KEY`; optional `YAPIT_SCRIPT_MODEL` selects the writer.
Only text is sent, not images, video frames or voice recordings.
There is one bounded estimated-duration revision, not unbounded retries.

## Input examples

These are examples of  generated file templates by YapIt.

### Narration script

Save this as `data\transcripts\trip.json`, then pass that path to `--script`.
Each beat contains spoken words, a visual direction and a pause in milliseconds.
This 30-word example estimates about 13 seconds at 150 words/minute; start with
`--seconds 15`, not 30. Actual synthesized duration still depends on the voice.

```json
{
  "title": "A calmer weekend plan",
  "beats": [
    {
      "text": "Want a weekend away without spending half of it in transit?",
      "visual": "Open on a packed bag and a nearby destination on a map.",
      "pause_ms": 180
    },
    {
      "text": "Pick one base, then group your stops around it.",
      "visual": "Show a simple route with nearby stops grouped together.",
      "pause_ms": 250
    },
    {
      "text": "Leave one afternoon unplanned. That's your buffer, not wasted time.",
      "visual": "A quiet cafe or a slow walk, then the route overview.",
      "pause_ms": 180
    }
  ]
}
```


## Worker and timing

The worker uses authenticated loopback HTTP, owner-only state, confined voice
assets and one non-queued generation lock. It rejects foreign hosts/origins
and disables proxy/redirect routing. Retain one model; close replaced/failed
adapters. It is not a hosted API, durable queue or machine-wide GPU lock.
Worker runtime/device wins over frontend device settings.

Cache stamps include checkpoint/config/reference changes and SDK version.
Cache bypass retains residency. Status separates load/request timings; a cache
hit is not an inference benchmark. Qwen codebook CUDA graphs default on for
supported CUDA configurations, with SDK fallback; `YAPIT_CUDA_GRAPHS=0` disables
them before model startup.

Each Script beat is one synthesis request. Five-millisecond edge ramps soften
joins without trimming or changing duration; requested pauses are appended,
not deduplicated against model-generated silence.
Audio-only fitting uses +/-5 seconds and 0.90-1.10x gates. Pipeline video passes
natural-duration tolerance so in-range speech stays 1.0x; out-of-range speech
requires revision. The encoded MP3 is checked again.
Timeline uses measured lengths/quiet-window sub-cuts and approximate word groups,
not forced alignment. Waveform intermediates and failure/status records remain.

## Quality limitations and tests

Signal gates do not establish identity, transcript correctness or naturalness.
Training overfit is not established by uneven pacing alone. Proposed improvements,
**not implemented**: optional natural audio-only timing, paragraph-level speech
chunks independent of visual beats, and complete effective-settings provenance.
Compare one change at a time on held-out text before claiming better speech.

```powershell
.\.venv\Scripts\python.exe -m pytest src\yapper\tests -q
```

Rapid units use synthetic PCM/fake SDKs. Media/worker subprocess workflows and
legacy enrolled fixtures are integration work, not evidence of real voice quality.
