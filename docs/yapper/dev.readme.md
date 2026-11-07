# Yapper development

Source is `src\yapper`; public imports are `yapit.yapper`. Documentation remains
under `docs`, not beside Python sources. See [usage](README.md) and the
[folder-level project layout](../dev.readme.md#flat-source-layout).

## Ownership

| File | Responsibility |
|---|---|
| `script_planner.py`, `script_prompt.txt` | Text/JSON loading, deterministic prompt composition and explicitly opted-in cloud writing |
| `speech.py` | `SpeechBackend` protocol and lazy native Qwen model/reference/checkpoint adapter |
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
target seconds, source name and an injected `SpeechBackend`. It normalizes plain
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

## Preserved policies and limitations

One Qwen backend and clone prompt are reused across a render. SDK imports/model
loading stay lazy; supplied-text parsing and prompt preparation need no model.
Automatic device selection logs CPU fallback at model load, not per beat;
the warning uses logging/stderr without contaminating CLI/MCP JSON stdout.
TTS returns validated mono 24 kHz PCM. Each beat receives its requested silence;
aggregate narration over 190 seconds is rejected without truncation.

Existing duration/tempo gates, raw/fitted audio retention, filename reservation,
provenance and failure logs are preserved. Tempo fitting scales inserted silence
too; exact pause preservation and natural-silence deduplication are not new
features. There is no per-word alignment, human-script emphasis analyser, automatic
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
