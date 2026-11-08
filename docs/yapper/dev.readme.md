# Yapper development

Source is `src\yapper`; public imports are `yapit.yapper`. Documentation remains
under `docs`, not beside Python sources. See [usage](README.md) and the
[folder-level project layout](../dev.readme.md#flat-source-layout).

## Ownership

| File | Responsibility |
|---|---|
| `script_planner.py`, `script_prompt.txt` | Text/JSON loading, deterministic prompt composition and explicitly opted-in cloud writing |
| `backends.py` | Backend protocols, lazy adapter selection, plugin registration and shared PCM validation |
| `speech.py` | Qwen-specific loading, reference/checkpoint handling and acceleration |
| `pocket.py` | Pocket TTS loading, reusable reference/voice state and CPU synthesis |
| `inference.py` | CUDA-graph replay of the fixed-length codebook predictor, with SDK fallback |
| `worker.py` | Authenticated loopback worker, one resident model, bounded private speech cache and load/request telemetry |
| `generation.py` | Narration workflow, reusable synthesis/encoding renderer and failure/status handling |
| `timeline.py` | Measured beat/pause scaling, quiet-window cuts and approximate phrase grouping |
| `cli.py`, `__main__.py`, `__init__.py` | CLI and public audio API |
| `tests` | Audio, planning, backend and narration-adapter tests using synthetic PCM/fake SDKs |

The old root-source `scripts.py`, `speech.py` and `timeline.py` are removed;
there are no legacy re-export modules. `script_prompt.txt` is loaded with
`importlib.resources.files("yapit.yapper")`, including in installed wheels.

## APIs and integration

```python
from yapit.yapper import generate_audio
from yapit.yapper.script_planner import build_prompt, load_script

result = generate_audio(
    "Take the slower route. Leave time to enjoy the view.",
    voice_name="arka",
    seconds=5,
)
```

`generate_audio` accepts `Script | str`, optional `Workspace`, voice/style,
target seconds, source name, an injected `SpeechBackend`, or `backend_name`/`model`.
It normalizes plain
text, validates the target, selects a profile and reserves outputs before calling
the renderer. It never calls the script-writing provider or Editor.

`generation.render_audio` takes a validated Script, selected Voice/reference
directory and pre-reserved `OutputArtifacts`, returning `(result, Timeline)`.
Root `Pipeline.reel` uses this lower-level API so audio and video share one job.
Root `audiogen` registers the module CLI's `generate` function directly, without
duplicating generation logic. CLI `narrate` and MCP narration use `generate_audio` directly.
Prompt/reel orchestration calls `script_planner` explicitly.
The old `yapit yapper generate` group is a hidden compatibility alias.
`python -m yapit.yapper generate` remains an optional standalone developer entry point.

Shared contracts remain in `models.py`, profile/style selection in
`enrollment\profiles.py` / `enrollment\styles.py`, and application folders/naming
in `storage.py`. Yapper reuses
`common_utils` for atomic JSON, native model loading, resampling, FFmpeg fitting,
WAV/MP3 encoding and decode validation. Do not duplicate these utilities here.

## TTS adapters

The renderer accepts a minimal `synthesize(text) -> np.ndarray` backend for
injection. Managed adapters implement `ModelBackend`: `load`, `close`,
`synthesize`, `cache_identity` and `describe`. Constructors stay lightweight;
SDK imports/model loading happen in `load`. Return validated mono float32
24 kHz PCM, converting the SDK's native rate inside the adapter.

`cache_identity` returns JSON-serializable model/version/conditioning information
and relevant asset stamps; exclude writing metadata. `describe` supplies model,
voice mode, device, dtype and inference-backend metadata without exposing SDK
internals to the worker. The worker also includes the selected adapter/model in
its cache key, closes replaced/failed models and retains one active model.
One-shot rendering closes adapters it creates; callers own injected adapters.

Register a trusted local adapter:

```python
from yapit.yapper import register_backend
from my_tts_adapter import MySpeech

register_backend("my-model", MySpeech)
```

Factories accept `(voice, voice_dir, *, model=None)`. In-memory registration is
process-local; install a package entry point to make it available in separate
CLI/worker processes:

```toml
[project.entry-points."yapit.tts"]
my-model = "my_tts_adapter:MySpeech"
```

Then select `--backend my-model --model MODEL_ID`. Plugins are trusted installed
Python code, not arbitrary imports supplied in HTTP requests. Unsupported
architectures need adapters; changing a Hugging Face ID alone cannot change SDKs.
Qwen's training/checkpoint validation stays in its training module.

Pocket optionally accepts `pocket_voice_state` in profile metadata, pointing to
an exported state inside the profile folder. Otherwise it uses `reference_audio`.
Its model/reference are retained between worker calls; custom configs should live
under the configured model/data root. Use immutable revisions for custom YAML
weights: cache identity stamps the config/reference, not every referenced weight.
Restart after changing those weights. The SDK interface/model-selection routes
are unit-tested with mocks. A separately authorized Pocket `english_2026-10`
Arka-reference CPU run produced the Puerto Princesa guide on November 8, 2026;
encoded-video/signal checks passed, but listening quality remains unapproved.
Cloning weights require publisher access approval and a local Hugging Face login.

## Preserved policies and limitations

When a worker state exists, `generation.render_audio` uses its local client.
The worker binds only 127.0.0.1 on an ephemeral port, requires a per-run bearer
token, rejects browser origins/foreign Host headers, disables proxy/redirect
routing and confines voice assets to the data root. Its state is owner-only
(Windows ACL or POSIX mode 0600); no token is printed.

One model is retained, with a non-queued single-generation lock. Switching
weights/speaker/reference reloads that model; writing metadata does not.
Checkpoint/config/reference timestamps and SDK version invalidate cached FLOAT
WAVs. Cache bypass retains the model. Startup is offline; device/dtype and
separate load/request timings are reported by `voice-worker-status`.
The worker's runtime controls the device even for CPU-only frontend clients;
`YAPIT_VOICE_WORKER=0` restores the frontend's one-shot device selection.
This is not a machine-wide GPU lock, hosted API or durable job queue.

CUDA narration uses codebook CUDA graphs by default. The first uncached request
captures the loop; later requests replay it without per-token Python dispatch.
Weights, precision and SDK sampling processors are unchanged. CPU, unsupported
batches/configurations and failed captures retain SDK generation.
Set `YAPIT_CUDA_GRAPHS=0` before starting the worker (or one-shot generation)
to disable this optimization. Worker health reports
`inference_backend` after synthesis; speech-cache hits are not inference benchmarks.

One Qwen backend and clone prompt are reused across a render. SDK imports/model
loading stay lazy; supplied-text parsing and prompt preparation need no model.
Automatic device selection logs CPU fallback at model load, not per beat;
the warning uses logging/stderr without contaminating CLI/MCP JSON stdout.
TTS returns validated mono 24 kHz PCM. A copy receives 5 ms onset/offset ramps,
leaving backend buffers and measured lengths unchanged. Each beat receives only
its requested silence; existing model-generated pauses are not deduplicated.
Audio-only aggregate narration over 190 seconds is rejected without truncation.
Reels also support the upper edge of the +/-33% range for a 180-second target
(239.4 seconds); Timeline permits up to 240 seconds.

`Pipeline.reel` passes `natural_duration_tolerance=brief.seconds * 0.33` to
`render_audio`/`fit_audio` for video jobs. In-range speech is retained at 1.0x;
out-of-range speech requires a script revision, not padding or stretching.
Prompt and optional cloud-planning acceptance use the same range.
Standalone `generate_audio` and narration-only Pipeline jobs retain the existing
+/-5-second and 0.90-1.10x fitting gates. Raw/fitted retention, filename
reservation, provenance and failure logs remain unchanged.
There is no per-word alignment, human-script emphasis analyser, automatic
weight training or production CosyVoice provider.

## Validation

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q src\yapper\tests
.\.venv\Scripts\ruff.exe check src tests scripts
.\.venv\Scripts\ruff.exe format --check src tests scripts
```

Tests own a package-scoped `conftest` so enrollment/root fixtures do not shadow
Yapper's synthetic speech helpers. They cover exact supplied words/default
pauses, duration failure, styles, non-overwriting outputs, fake Qwen SDK calls,
text/TXT/JSON CLI inputs, cloud opt-in and bounded planning revision.
Root CLI/MCP/reel tests remain cross-module integration tests.
