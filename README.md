# YapIt

Turn reviewed text into narration in your enrolled voice and a portrait editing
guide. Rescript extracts/transcribes local media; Yapper plans scripts and
synthesizes audio; Editor renders duration-coloured placeholders; training is a
separate, explicitly requested operation.

**The video is an editing guide, not a finished reel.** Replace its numbered
solid-colour placeholders with your footage. Recognition, voice similarity,
delivery, timing and captions require human review.

## Documentation index

All project Markdown except this README lives in `docs`. User guides cover each
module's operations; developer guides cover its implementation and tests.

| Document | Dev guide | Purpose |
|---|---|---|
| [Project README](README.md) | [Project development](docs/dev.readme.md) | Setup, command inventory, configuration, folder structure and integration |
| [Enrollment](docs/enrollment/README.md) | [Enrollment development](docs/enrollment/dev.readme.md) | Model-free recording, profiles and sample curation |
| [Rescript](docs/rescript/README.md) | [Rescript development](docs/rescript/dev.readme.md) | Audio extraction, local ASR and reviewable scripts |
| [Yapper](docs/yapper/README.md) | [Yapper development](docs/yapper/dev.readme.md) | Script planning, text/JSON narration and measured timing |
| [Editor](docs/editor/README.md) | [Editor development](docs/editor/dev.readme.md) | Duration-coloured placeholders, captions and video rendering |
| [Training](docs/training/README.md) | [Training development](docs/training/dev.readme.md) | Explicit fine-tuning and reviewed checkpoint selection |
| [MCP](docs/mcp/README.md) | [MCP development](docs/mcp/dev.readme.md) | Local stdio tools, client configuration and protocol integration |
| [Audio design](docs/audio-design.md) | - | Design rationale and listening-validation requirements |
| [Library boundaries](docs/module-architecture.md) | - | Current logical modules and proposed independently distributed providers |
| [Expressive voice research](docs/analysis.md) | - | Historical model study; not a current CLI or implementation guide |

## Setup

Use Python 3.11 or 3.12. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
# Only if you do not already have .env:
Copy-Item .env.example .env
.\.venv\Scripts\yapit doctor
```

Install `.[voice]` for narration, `.[rescript]` for transcription, `.[mcp]` for
the MCP server, and `.[dev]` for tests/build tooling. Cloud scripting and actual
fine-tuning have separate optional dependencies and permission/hardware requirements.
Model weights are provisioned separately; installation alone is not offline-ready.
See the developer guide for compatible runtimes, offline setup and troubleshooting.

For isolated Windows CPU/GPU runtimes without activation:

```powershell
.\scripts\setup-mode.cmd
# Or choose explicitly:
.\scripts\setup-mode.cmd cpu
.\scripts\setup-mode.cmd gpu
.\scripts\yapit-mode.cmd cpu doctor
.\scripts\yapit-mode.cmd gpu --help
```

Omitting the setup mode defaults to `auto`: an NVIDIA driver probe selects GPU,
otherwise setup selects CPU and logs the reason. Explicit GPU dispatch requires
working CUDA availability and does not silently fall back to CPU.
The launchers share configured data/model roots and change only the child environment.
Setup downloads dependencies/tools, not model weights.

Examples below use `yapit`. Without activating your environment, substitute
`.\.venv\Scripts\yapit` or `.\scripts\yapit-mode.cmd cpu` / `gpu`.

## Available commands

Operations are flat: **`yapit audiogen`**, not `yapit yapper generate`.
Run `yapit COMMAND --help` for arguments, options and defaults.
Model annotations include locally loaded models, not just cloud APIs.
Qwen3-TTS is the local speech language model used for narration/training,
not a general-purpose chat model. Whisper is listed separately as local ASR.

| Command | Functionality |
|---|---|
| `yapit doctor` | Report dependencies, FFmpeg and CUDA availability; no speech-model download |
| `yapit profiles` | List saved profile names, including incomplete profiles with saved metadata |
| `yapit enroll-app` | Open the Windows microphone enrollment UI |
| `yapit voice-enroll` | Import/merge an existing recordings manifest; at least five distinct clips |
| `yapit voice-style-prompt` | Get an expressive reference passage and recording directions |
| `yapit enroll-style` | Save your verified expressive recording to a validated existing profile; consent defaults to true |
| `yapit extract_audio` | Extract video audio, convert decodable audio or copy supported audio |
| `yapit transcript` | Transcribe into Script JSON/timing; cached/offline by default (local ASR model required: Whisper) |
| `yapit prompt` | Prepare a script-writing prompt from a brief and optional local image/video frames |
| `yapit audiogen` | Generate narration/timing from exactly one of `--text` or `--script` (local speech LLM required: Qwen3-TTS) |
| `yapit narrate` | File-only narration; voice is an optional positional argument (local speech LLM required: Qwen3-TTS) |
| `yapit reel` | Generate narration/edit guide; without a script, return a prompt (local speech LLM required for narration: Qwen3-TTS) |
| `yapit voice-train` | Linux/WSL2 CUDA fine-tuning; `--resume DIRECTORY` continues weights with a new optimizer (local speech LLM required: Qwen3-TTS) |
| `yapit voice-select-checkpoint` | Manually select a reviewed CustomVoice checkpoint for inference |

Root options are `--help`, `--install-completion` and `--show-completion`.
`audiogen --voice NAME` selects a profile; narration/reel default to `arka`.
`reel --auto-script` is the optional cloud-writing path, not a separate command.

### Typical workflow

```powershell
yapit enroll-app
yapit profiles
yapit extract_audio .\trip.mp4
yapit transcript .\data\audio\arka.trip.wav
yapit prompt --topic "A relaxed weekend route" --seconds 30
# Review the recognized/written words and save a Script JSON or UTF-8 text file.
yapit audiogen --script .\data\transcripts\trip.json --voice arka --seconds 30
yapit reel arka --script .\data\transcripts\trip.json --seconds 30
```

Inline words work too:

```powershell
yapit audiogen --text "Take the slower route. Leave time to enjoy the view." --voice arka --seconds 5
```

Supplied narration text is not rewritten by an LLM. Plain text gets default
sentence/paragraph pauses; Script JSON supports explicit `pause_ms` and visual
suggestions. Targets are 5-180 seconds, with the existing +/-5-second tolerance
and bounded 0.90-1.10x tempo fitting. Final timing is measured, not guaranteed by
text estimates. Enrollment/reference cloning does not train model weights.
Transcription defaults to cached/local weights; explicit `--offline` is redundant.
Use `--no-offline` to permit model downloads, not hosted recognition. No online
transcription provider is supported.

### Compatibility and isolated module entry points

Old grouped commands remain hidden from root help for existing scripts:
`yapit yapper generate`, `yapit rescript extract_audio`,
`yapit rescript transcript`.
`yapit voice-use-checkpoint` remains a hidden inference-selection alias, not
training resumption. Use the flat forms above for new commands.
`voice-style-enroll` was renamed to `enroll-style`; update old calls.

Optional standalone developer entry points are
`python -m yapit.yapper generate`,
`python -m yapit.rescript.cli extract_audio|transcript` and
`python -m yapit.enrollment enroll-app|voice-enroll|profiles`.
They are not required for ordinary usage. Editor is reached through `reel`,
MCP `render_timeline` or Python; training through the two voice-training/checkpoint
commands or Python. Neither has a standalone CLI group.

## MCP and helper commands

`yapit-mcp` (or `python -m yapit.mcp`) starts the local stdio MCP server.
Install `.[mcp]` and adapt `src\mcp\examples\mcp.config.json` with absolute paths.
MCP-specific implementation, tests and the example configuration live in
`src\mcp`; see the [MCP guide](docs/mcp/README.md).
The old `python -m yapit.server` path is replaced by `python -m yapit.mcp`.
MCP names are protocol tools, **not** additional `yapit` CLI commands:

| MCP tool / prompt | Purpose |
|---|---|
| `prepare_script_prompt` | Prepare a brief/schema prompt with local frame paths |
| `enroll_voice` | Enroll a confined recordings manifest |
| `generate_narration` | Synthesize a supplied Script without video |
| `generate_reel` | Orchestrate scripting/narration/edit guide |
| `render_timeline` | Render an existing matching timeline/audio pair |
| `get_job_status` | Read status by the returned output name |
| `reel_script` (prompt) | Return reusable script-writing instructions |

All MCP file inputs must stay inside the configured workspace or data root.
Calls are synchronous; there is no durable job queue or cross-process GPU lock.

| Helper command | Purpose |
|---|---|
| `.\scripts\setup-mode.cmd [auto\|cpu\|gpu]` | Provision an isolated runtime; default auto selection; optional `--wheelhouse PATH` |
| `.\scripts\yapit-mode.cmd cpu` / `gpu` | Dispatch any flat YapIt command; `python <args>` or `mcp` are launcher alternatives |
| `& ".\scripts\Start Enrollment.cmd"` | Open enrollment in CPU mode; optional `gpu` argument |
| `python .\scripts\cache-feed-wheel.py WHEEL_BASENAME` | Cache one hash-verified wheel from the configured approved feed; optional `--folder PATH` |
| `python .\scripts\cosyvoice-trial.py --script PATH` | Experimental comparison; optional `--profile`/`--instruction`, separate worker/models required; currently paused |

`setup-mode.py` and `run-mode.py` are Python backends for the CMD wrappers,
not additional product operations. `hatch-build.py` is a build hook and the
CosyVoice worker is an internal subprocess adapter. Their details belong in the
developer guide; no production CosyVoice backend is implied.

## Files, privacy and review

| Location | Content |
|---|---|
| `data\<profile>` | Accepted enrollment recordings/profile metadata and training checkpoints |
| `data\audio` | Imported/extracted source audio; external transcription archives are under `imports` |
| `data\transcripts` | Reviewed scripts, ASR sidecars, captions and training JSONL |
| `data\generated_audio` | Final WAV/MP3 and raw/fitted synthesis intermediates |
| `data\generated_video` | Completed edit-guide MP4s |
| `workspace\outputs` | Prompts, measured timelines, provenance/status, logs and temporary rendering files |
| `workspace\enrollment` | Resumable drafts/raw takes; removed after successful enrollment |
| `models` | Shared model assets/cache, configurable with `YAPIT_MODELS_DIR` |

Set absolute `YAPIT_DATA_DIR` / `YAPIT_WORKSPACE` for alternate launch locations.
Generation/Rescript default to data beside the workspace; enrollment independently
defaults to `.\data`. The data override is shared by all modules.

Processing is local unless cloud script writing is explicitly enabled. Downloads
need network access; manual sharing with an assistant is a separate decision.
Keep `.env`, credentials, weights and private recordings out of commits.
`data\arka` is eligible for Git but is never automatically staged/uploaded.
Source archives exclude the repository's `data` directory; selected tracked
fixtures under `tests\data` are a separate test-only exception.

Review recognized words and generated audio before use. Passing model-free tests
does not establish voice quality, correct recognition or exact word alignment.
