# YapIt

Generate local narration in your enrolled/trained voice and a portrait edit guide.
**Videos contain numbered placeholders, not travel footage.** Replace them with
your clips. Music is opt-in; training is always separate.

## Documentation index

| Document | Dev guide | Purpose |
|---|---|---|
| [Project](README.md) | [Development](docs/dev.readme.md) | Configuration and integration |
| [Enrollment](docs/enrollment/README.md) | [Development](docs/enrollment/dev.readme.md) | Recordings and voice profiles |
| [Rescript](docs/rescript/README.md) | [Development](docs/rescript/dev.readme.md) | Extraction and transcription |
| [Yapper](docs/yapper/README.md) | [Development](docs/yapper/dev.readme.md) | Scripts, narration and warm worker |
| [Musicgen](docs/musicgen/README.md) | [Development](docs/musicgen/dev.readme.md) | CC0 music acquisition |
| [Editor](docs/editor/README.md) | [Development](docs/editor/dev.readme.md) | Mixing, captions and edit guides |
| [Training](docs/training/README.md) | [Development](docs/training/dev.readme.md) | Fine-tuning and checkpoint selection |
| [MCP](docs/mcp/README.md) | [Development](docs/mcp/dev.readme.md) | Local tools and client setup |

## Setup

Python 3.11/3.12, PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\yapit --help
```

Install `.[voice]`, `.[rescript]` or `.[mcp]` for the corresponding module.
Weights/tokenizers are provisioned separately. Windows CPU/GPU setup and offline
instructions are in the [developer guide](docs/dev.readme.md).
Without activation, replace `yapit` below with `.\.venv\Scripts\yapit` or
`.\scripts\yapit-mode.cmd gpu`.

## Available commands

Commands are flat; use `yapit COMMAND --help`. Narration uses local Qwen3-TTS;
transcription uses local Whisper.

<details>
<summary>All commands</summary>

| Command | Functionality |
|---|---|
| `yapit doctor` | Dependencies and CUDA status |
| `yapit profiles` | Saved voice profiles |
| `yapit enroll-app` | Microphone enrollment UI |
| `yapit voice-enroll` | Import verified recordings |
| `yapit voice-style-prompt` | Expressive recording directions |
| `yapit enroll-style` | Save a style reference |
| `yapit extract_audio` | Extract, convert or copy audio |
| `yapit transcript` | Offline transcription to Script JSON |
| `yapit prompt` | Script-writing prompt |
| `yapit audiogen` | Text/JSON narration and timing |
| `yapit narrate` | File-only narration |
| `yapit reel` | Narration and edit guide |
| `yapit videogen` | Render existing audio/timing |
| `yapit music-plan` | Local music brief |
| `yapit music-search` | CC0 music discovery |
| `yapit music-download` | Source-verified CC0 download |
| `yapit voice-worker` | Resident model and speech cache |
| `yapit voice-worker-status` | Device, readiness and timings |
| `yapit voice-worker-stop` | Release the resident model |
| `yapit voice-train` | Explicit Linux/WSL2 fine-tuning |
| `yapit voice-select-checkpoint` | Select reviewed trained weights |

</details>

### Typical workflow

In a separate GPU terminal: `yapit voice-worker arka`. Then:

```powershell
yapit reel arka --script .\data\transcripts\trip.json --seconds 30
yapit reel arka --script .\data\transcripts\trip.json --music .\data\background_music\TRACK.mp3
yapit voice-worker-status
yapit voice-worker-stop
```

The optional worker stays loaded between requests and caches unchanged speech.
Its runtime determines CPU/GPU usage; clients need only the core dependencies.
See [Yapper](docs/yapper/README.md) for cache bypass and offline setup.

Reviewed text is not rewritten. Videos accept **+/-33%** of the target, including
the default 0.75s closing hold; speech is not stretched/padded to fit.
Music-only fades, ducking and separate 0-100 gains are available.
`videogen` reuses existing narration/timing without synthesis.

## MCP and helper commands

`yapit-mcp` starts the local stdio server; see [MCP setup](docs/mcp/README.md).
Runtime launchers, compatibility aliases and experimental helpers are documented
in the [developer guide](docs/dev.readme.md).

## Files, privacy and review

| Location | Content |
|---|---|
| `data\<profile>` | Accepted enrollment recordings/profile metadata and training checkpoints |
| `data\audio` | Imported/extracted source audio; external transcription archives are under `imports` |
| `data\transcripts` | Reviewed scripts, ASR sidecars, captions and training JSONL |
| `data\generated_audio` | Final WAV/MP3 and raw/fitted synthesis intermediates |
| `data\generated_video` | Completed edit-guide MP4s |
| `data\background_music` | CC0 music named `emotion_videotype_songname_or_id.mp3`, with source/license/plan JSON |
| `workspace\outputs` | Prompts, measured timelines, provenance/status, logs and temporary rendering files |
| `workspace\enrollment` | Resumable drafts/raw takes; removed after successful enrollment |
| `models` | Shared model assets/cache, configurable with `YAPIT_MODELS_DIR` |
| `.repo-stats\<module>` | Git-ignored function/test timings and outcomes; no narration or credentials |

Use `YAPIT_DATA_DIR`, `YAPIT_WORKSPACE` and `YAPIT_MODELS_DIR` to override roots.
Worker state/cached speech live privately under `data\.runtime`.
Keep credentials, profiles, recordings, weights and generated media out of commits.
Review transcripts, pronunciation, voice quality and music before publishing.
