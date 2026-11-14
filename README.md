# YapIt

**YapIt is profile-based personalized voice generation for engaging travel and
product reels.** Create a voice profile from your recordings, choose a TTS model,
and reuse your voice for narration and portrait edit guides.

**Videos contain numbered placeholders, not travel footage.** Replace them with
your own clips. Music is opt-in; voice training is always a separate step.

## Demo

[**Watch a YapIt generated demo (MP4)**](docs/res/generated_demo.puerto_princesa.pocket_2026_10.3.guide.mp4)

A generated narration/edit-guide example, not a finished footage reel.
The demo uses PocketTTS backend; the default installer below
installs Qwen. Your own voice, script and selected model determine the result.

## Install locally on Windows

You need **64-bit Python 3.12** with the Windows Python launcher (`py`) and
Tkinter, **Git**, an internet connection for packages/first-use model downloads,
and enough disk space for an isolated runtime and model weights (several GB).
A microphone is needed only for recording your voice. No API key or cloud
subscription is needed for local narration.

In PowerShell, clone this repository into a folder you own:

```powershell
git clone https://github.com/Arka95/yapit.git
Set-Location .\yapit
py -3.12 --version
.\scripts\setup-mode.cmd
```

The installer already checks for an NVIDIA GPU with a bounded `nvidia-smi` probe:

| Detection | Installation |
|---|---|
| NVIDIA GPU reported by the driver | GPU runtime in `.venv-gpu` |
| No NVIDIA GPU, missing driver tool, or failed/timed-out probe | CPU runtime in `.venv-cpu`, with a visible fallback message |

Setup saves its selected CPU/GPU runtime in your local `.env`. It installs
the speech, transcription and MCP dependencies, matching
PyTorch/torchaudio wheels, and checksum-verified SoX. It preserves any existing
`.venv`; it does **not** download speech model weights, train a voice, install
NVIDIA drivers, or configure your agent.

Detection is not a GPU compatibility guarantee. GPU mode needs a driver/device
compatible with the pinned CUDA 12.8 runtime, and validates CUDA before running.
It fails rather than silently switching devices. To choose a mode explicitly:

```powershell
.\scripts\setup-mode.cmd cpu
# Or, with a compatible NVIDIA GPU:
.\scripts\setup-mode.cmd gpu
```

Everyday commands need no mode argument:

```powershell
.\yapit.cmd doctor
.\yapit.cmd --help
```

To change the default later, set it once:

```powershell
.\yapit.cmd set-mode gpu
# Or:
.\yapit.cmd set-mode cpu
```

`set-mode` saves `YAPIT_MODE=gpu` or `cpu` in the existing project `.env`,
preserving other settings. The next command uses that runtime. It does not
install dependencies or move an already-running voice worker; install the
chosen runtime first and restart a worker when changing its device.
An existing process environment variable `YAPIT_MODE` overrides the saved value.
Without either setting, the default is CPU; there is no automatic runtime
switching after setup.

PowerShell requires `.\yapit.cmd` when launching from this folder. In Command
Prompt, or with the clone folder on `PATH`, you can use `yapit.cmd doctor`
directly. The launcher does not modify your global `PATH`.

Python commands are also supported from an activated YapIt environment:

```powershell
python .\scripts\run-mode.py default doctor  # Honor the saved CPU/GPU runtime.
python -m yapit.cli doctor                  # Use this interpreter directly.
python -m yapit.cli enroll-voice
python -m yapit.mcp
```

Direct module calls do not switch environments or apply launcher device/DLL
settings. There is no top-level `python -m yapit` entry point.

`doctor` reports dependencies, FFmpeg and CUDA availability; it does not download
models or prove generation quality. CPU is supported, but no real-time generation
speed is promised. Windows is the documented customer setup; this installer and
the microphone UI are not cross-platform installers.

Python 3.11 is also supported by the Python setup script. Without Python 3.12,
use `py -3.11 .\scripts\setup-mode.py` instead of the `.cmd` setup wrapper.
Advanced/offline installation and troubleshooting:
[developer guide](docs/dev.readme.md#isolated-cpugpu-runtimes-on-windows).

## Create your first narration and reel

Run these commands from the cloned repository.
Create **your own** profile; `testuser` is the repository's existing demo/test
identity, not your personal voice. A name change does not anonymize recordings.

### 1. Record your voice

```powershell
.\yapit.cmd enroll-voice
```

Create a profile such as `myvoice`, choose your microphone, record the displayed
passages and review each take. Save **at least five distinct recordings**, then
finish enrollment. This creates a reusable reference profile; it does not train
model weights. No recording files or manifest are required;
see [enrollment](docs/enrollment/README.md).

```powershell
$voice = "myvoice" # Must match the profile you created.
.\yapit.cmd profiles
```

### 2. Review a script

Copy the [script example](docs/yapper/dev.readme.md#narration-script) to
`data\transcripts\trip.json`, or prepare a new script with your agent. This command
generates Script JSON with explicit cloud opt-in and credentials:

```powershell
.\yapit.cmd generate-script "A calmer weekend trip" --seconds 15
```

Save the reviewed Script JSON in `data\transcripts`.
Narration preserves the supplied words rather than rewriting them.

### 3. Generate audio or an edit guide

```powershell
# Audio only:
.\yapit.cmd generate-audio --voice $voice --script .\data\transcripts\trip.json --seconds 15

# Audio plus vertical video editing guide:
.\yapit.cmd generate-reel --voice $voice --script .\data\transcripts\trip.json --seconds 15
```

The default backend is Qwen. The first generation can download its model assets
if missing; package installation alone is not an offline-ready setup. Review the
download requirements before starting. Later offline use requires the complete
model/tokenizer cache; see [model setup](docs/dev.readme.md#configuration-and-dependencies).
Pocket is an optional backend setup, not automatically
chosen by GPU detection; see [backend choices](docs/yapper/README.md#choose-a-model).

Commands return JSON with output paths. Narration is saved in
`data\generated_audio`, guide videos in `data\generated_video`, scripts/captions
in `data\transcripts`, and job metadata/timelines in `workspace\outputs`.
Listen to the result and replace numbered placeholders with your own footage.
For audio only, omit `--seconds` to keep natural duration with no total duration
cap or target check. An explicit target uses +/-33% validation without stretching,
padding or truncation. Reels still default to a 30-second target.
If natural speech misses an explicit target, adjust the script.

### 4. Optional music and repeated generation

Music is off by default. Supply an existing track explicitly:

```powershell
.\yapit.cmd generate-reel --voice $voice --script .\data\transcripts\trip.json --seconds 15 --music .\data\background_music\TRACK.mp3
```

For repeated requests, after provisioning model assets, run
`.\yapit.cmd voice-worker $voice` in a separate terminal.
Use `voice-worker-status` to check readiness and `voice-worker-stop` to release
it. The worker keeps one model loaded and caches unchanged speech; see
[Yapper](docs/yapper/README.md) for regeneration/cache controls.

## Use YapIt from an AI agent

YapIt has **one local stdio MCP server**, with five tools.
Use an agent/desktop client that can launch a **local stdio server on your
machine**. A cloud-only client without local execution cannot use this server
directly.

Start with [the client configuration example](src/mcp/examples/mcp.config.json).
Replace all example paths with your clone's absolute paths. Its
`mcpServers` envelope is client-specific:
adapt it to your client's configuration format. The launcher runs
`.\yapit.cmd mcp`, using your saved runtime; it waits for an MCP client rather than
opening an interactive console.

| MCP tool | What your agent can do |
|---|---|
| `generate_script` | Generate YapIt Script JSON from `prompt` and optional `seconds` (default 30); requires cloud opt-in |
| `generate_audio` | Generate voice audio from reviewed Script JSON; defaults to `testuser` and natural duration; optional target uses +/-33% |
| `extract_script` | Extract reviewable Script JSON and source timings from local audio/video |
| `enroll_voice` | Open the enrollment app to choose a profile, record and review your voice; no inputs |
| `generate_reel` | Generate a vertical placeholder edit guide from exactly one `script`, `audio_path` or `prompt` |

Tell your agent your own profile name explicitly. Files passed to MCP tools
must be within the configured data/workspace roots. Calls finish synchronously;
there is no background job queue. Music search,
worker management and training currently have CLI commands, not MCP tools.
See [MCP setup](docs/skills/README.md) for execution limits and configuration.

## Troubleshooting

| Problem | What to check |
|---|---|
| `py -3.12` is unavailable | Install Python 3.12 with its launcher/Tkinter, or use the Python 3.11 setup command above |
| NVIDIA hardware exists but CPU was selected | Check driver installation and `nvidia-smi`; explicit GPU setup still requires working CUDA |
| GPU launcher rejects CUDA | Fix driver/runtime compatibility, or install CPU and run `.\yapit.cmd set-mode cpu`; it is not silently selected |
| Changing mode appears to have no effect | Check for a process-level `YAPIT_MODE` override and restart an already-running voice worker |
| Package/tool download fails | Keep TLS/hash checks enabled; see the verified-wheel/offline setup in the developer guide |
| No personal voice/profile found | Finish enrollment and pass its actual name; do not assume the demo profile is your voice |
| Worker cannot find model assets | Provision the complete model/tokenizer before worker startup; workers start offline |
| MCP cannot open a file | Use absolute configured roots and put the input under data/workspace |
| MCP client sees invalid protocol output | Launch the `mcp` command, not the normal CLI; stdout belongs to MCP messages |

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
| [MCP](docs/skills/README.md) | [Development](docs/mcp/dev.readme.md) | Local tools and client setup |
| [Licensing](docs/licensing.md) | - | Commercial use, model attribution and future skill licensing |

## Available commands

Commands are flat. Below, `yapit COMMAND` means
`.\yapit.cmd COMMAND`; append `--help` for options.
Narration uses selectable TTS backends (Qwen by default); transcription uses
local Whisper.

The canonical workflows are `generate-audio`, `generate-script`, `extract-script`,
`generate-reel` and `enroll-voice`. MCP tools and the owning modules' Python
functions use the same names with underscores. Each workflow has one command;
see [workflow inputs and examples](docs/skills/README.md#mcp-setup-and-workflows).

### All commands

| Command | Functionality |
|---|---|
| `yapit set-mode` | Save the runtime choice (`cpu` or `gpu`) for future calls |
| `yapit doctor` | Dependencies and CUDA status |
| `yapit profiles` | Saved voice profiles |
| `yapit voice-style-prompt` | Expressive recording directions |
| `yapit enroll-style` | Save a style reference |
| `yapit extract_audio` | Extract, convert or copy audio |
| `yapit generate-script` | Generate Script JSON from a text prompt |
| `yapit generate-audio` | Speak supplied text/Script JSON |
| `yapit extract-script` | Media to reviewable inline Script JSON and saved transcripts |
| `yapit generate-reel` | Exactly one `--script`, `--audio-path` or `--prompt` to an edit guide |
| `yapit enroll-voice` | Open the microphone enrollment app; no inputs |
| `yapit music-plan` | Local music brief |
| `yapit music-search` | CC0 music discovery |
| `yapit music-download` | Source-verified CC0 download |
| `yapit voice-worker` | Resident model and speech cache |
| `yapit voice-worker-status` | Device, readiness and timings |
| `yapit voice-worker-stop` | Release the resident model |
| `yapit voice-train` | Explicit Linux/WSL2 fine-tuning |
| `yapit voice-select-checkpoint` | Select reviewed trained weights |

Reviewed text is not rewritten. Videos accept **+/-33%** of the target, including
the default 0.75s closing hold; speech is not stretched/padded to fit.
Music-only fades, ducking and separate 0-100 gains are available.
`generate-reel --audio-path` reuses existing narration without synthesis.

## Files, privacy and review

| Location | Content |
|---|---|
| `data\<profile>` | Accepted enrollment recordings/profile metadata and training checkpoints |
| `data\audio` | Imported/extracted source audio; external transcription archives are under `imports` |
| `data\transcripts` | Reviewed scripts, ASR sidecars, captions and training JSONL |
| `data\generated_audio` | Final WAV/MP3 and raw synthesis intermediates |
| `data\generated_video` | Completed edit-guide MP4s |
| `data\background_music` | CC0 music named `emotion_videotype_songname_or_id.mp3`, with source/license/plan JSON |
| `workspace\outputs` | Prompts, measured timelines, provenance/status, logs and temporary rendering files |
| `workspace\enrollment` | Resumable drafts/raw takes; removed after successful enrollment |
| `models` | Shared model assets/cache, configurable with `YAPIT_MODELS_DIR` |
| `.repo-stats\<module>` | Git-ignored function/test timings and outcomes; no narration or credentials |

Use `YAPIT_DATA_DIR`, `YAPIT_WORKSPACE` and `YAPIT_MODELS_DIR` to override roots.
Worker state/cached speech live privately under `data\.runtime`.
Keep credentials, model weights and generated media out of commits. Commit profile
data or audio samples only when intentionally selected; testuser's checkpoints are ignored.
Review transcripts, pronunciation, voice quality and music before publishing.

## License

YapIt's source code and project documentation are [MIT licensed](LICENSE).
Commercial use is allowed, but MIT also permits others to reuse and redistribute
the covered code. Models, recordings and generated media have separate rights
and conditions; see [licensing and commercial use](docs/licensing.md).
Pocket TTS demo backend: **Kyutai Pocket TTS**.
Any future restricted paid skill must have its own explicit licensing
boundary; marketplace terms do not remove existing open-source permissions.
