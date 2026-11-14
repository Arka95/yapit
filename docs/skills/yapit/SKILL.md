---
name: yapit
description: >-
  Set up YapIt locally on Windows, enroll a voice through its recording app,
  create or extract narration scripts, and generate personalized audio or
  portrait reel edit guides using YapIt's five local MCP tools.
summary: Local voice narration and portrait edit guides from reviewed scripts or existing audio.
version: "0.1.0"
tags:
  - audio
  - video
  - narration
  - local-first
keywords:
  - YapIt
  - voice enrollment
  - MCP
  - reel editing
docs_url: https://github.com/Arka95/yapit#readme
---

# YapIt

Use this skill when the user wants to set up YapIt, record their voice, turn
reviewed words into narration, extract a script from media, or create a portrait
edit guide. An edit guide contains numbered placeholders, not finished footage.

## Requirements

YapIt runs on the customer's machine. This skill provides instructions, not a
hosted generation service, installed application, model weights or voice profile.
The agent needs either access to YapIt's local stdio MCP server or a local
terminal for setup/CLI use. A cloud-only agent cannot launch this local server.

The documented setup requires Windows, 64-bit Python 3.12 with the `py` launcher
and Tkinter, Git, and several GB for dependencies/models. Packages and uncached
model assets need internet access. CPU is supported; GPU requires compatible
NVIDIA hardware/drivers. Recording requires a microphone and an interactive
Windows desktop.

Local narration does not require an API key. YapIt's `generate_script` tool and
the prompt-based reel pipeline use a cloud writer and require separate opt-in,
the cloud extra and `OPENAI_API_KEY`. Existing-audio reels need cached local
speech-recognition assets, but no enrolled voice or new speech generation.
The default installer uses Qwen for narration; Pocket is an optional backend,
not an automatic GPU/CPU fallback.

## Set up only if needed

Reuse an existing installation rather than cloning over it. For a new install,
run these commands in the customer's chosen parent directory:

```powershell
git clone https://github.com/Arka95/yapit.git
Set-Location .\yapit
py -3.12 --version
.\scripts\setup-mode.cmd
.\yapit.cmd doctor
```

Setup selects an isolated CPU/GPU runtime and saves it in the local `.env`.
It installs dependencies, not speech model weights. `doctor` checks dependencies
and device availability; it does not prove model readiness or audio quality.
If Python 3.12 is unavailable, the supported Python 3.11 alternative is
`py -3.11 .\scripts\setup-mode.py`.

Explicit runtime installation uses `.\scripts\setup-mode.cmd cpu` or `gpu`.
After that runtime is installed, `.\yapit.cmd set-mode cpu` or `gpu` changes the
saved choice. A process-level `YAPIT_MODE` overrides the saved value. Restart
existing servers/workers after changing mode; GPU failures do not silently
switch to CPU.

### Connect the local MCP server

Adapt this example to the customer's client configuration format. Replace every
`C:\\YapIt` with the actual clone directory; it is an example, not a required
installation location. Merge with existing client settings.

```json
{
  "mcpServers": {
    "yapit": {
      "command": "C:\\Windows\\System32\\cmd.exe",
      "args": ["/d", "/c", "C:\\YapIt\\yapit.cmd", "mcp"],
      "env": {
        "YAPIT_WORKSPACE": "C:\\YapIt\\workspace",
        "YAPIT_DATA_DIR": "C:\\YapIt\\data",
        "YAPIT_ALLOW_CLOUD": "false"
      }
    }
  }
}
```

The client launches the server over stdio. Running `.\yapit.cmd mcp` in a
terminal alone just waits for an MCP client. Direct `python -m yapit.mcp`
uses that interpreter, not the launcher's saved runtime selection.

After connecting, inspect the client's available tools. YapIt exposes exactly
`generate_script`, `generate_audio`, `extract_script`, `generate_reel` and
`enroll_voice`. Client-specific prefixes may precede those names. There is no
job-status, profile-listing, training or helper-prompt MCP tool.

## Pick the requested workflow

| User input or goal | MCP tool | Important choice |
|---|---|---|
| Record a voice | `enroll_voice` | No arguments; finish in the app |
| Write a script from text | `generate_script` | Cloud writer; review returned JSON |
| Extract words from media | `extract_script` | Local ASR, offline by default |
| Speak reviewed Script JSON | `generate_audio` | Pass the user's actual profile name |
| Make an edit guide | `generate_reel` | Exactly one script, audio path or prompt |

Use the requested workflow, not every step by default. Existing-audio reels do
not need enrollment, a cloud writer or `generate_audio`. Approved scripts do
not need another writing call. Do not add images/frames to script-writing calls.

### Enroll a voice

Call `enroll_voice` with `{}`. The app handles profile selection/creation,
microphone choice, displayed passages, recording, playback and completion.
The user must save at least five distinct recordings and finish in the app.
Do not request an MP3, transcript or manifest as tool input.

The return value is `status: launched` with a process ID, not completed
enrollment or proof the window stayed open. Wait for the user to finish before
using that profile. `.\yapit.cmd profiles` can list saved profiles from a local
terminal; it is not an MCP tool. Enrollment does not train model weights.

### Create or supply a script

Call `generate_script` with a text `prompt` and optional `seconds`:

```json
{"prompt": "A calm 30-second introduction to a weekend coastal walk.", "seconds": 30}
```

The default is 30 seconds; the supported target range is 5-180. The tool returns
Script JSON, not audio. Enable `YAPIT_ALLOW_CLOUD=true` and provide
`OPENAI_API_KEY` in the server's environment only when using cloud writing.
Install `.[cloud]` into the selected runtime if missing, and restart the server
after changing its environment. Do not put credentials in the skill or examples.

Alternatively, the user or agent can prepare Script JSON without calling
YapIt's cloud writer:

```json
{
  "title": "Coastal walk",
  "beats": [
    {
      "text": "The coast feels different when you slow down. Today, we are leaving the crowded roads behind and following a quiet path beside the water. There is no packed itinerary, just fresh air, a few beautiful views, and time to notice the small things along the way. Along the route, we will stop for a simple lunch, watch the waves roll in, and find a comfortable spot to enjoy the afternoon.",
      "visual": "Wide coastal walking shot",
      "pause_ms": 180
    }
  ]
}
```

Review words, facts, names and intended duration before narration. A Script
needs a title and 1-40 beats, with at most 650 words overall. Each beat has text,
an optional visual description, and an optional pause of 0-1200 milliseconds.
Visual descriptions are editing guidance, not generated images.

### Generate narration

Call `generate_audio` with the reviewed object under `script`, the actual
enrolled profile under `voice`, and optional `seconds`. For example, use the
script above with `voice: "myvoice"` and `seconds: 30`. Pass JSON, not a script
filename or a writing prompt.

The defaults are `voice: "testuser"`, natural duration, and `style: "default"`.
`testuser` is a demo identity, not the customer's voice; use their profile name
explicitly. Named styles require separately enrolled reference recordings.
Supplied words are not rewritten. Omit `seconds` for natural audio with no total
duration cap, fitting or duration-target check. An explicit target (5-180 seconds)
validates within +/-33%; speech is not stretched, padded or truncated.

The result includes MP3/WAV, saved script, timeline, provenance and status-file
paths. It creates no video. First use may download uncached speech assets.
Listen to the result; successful generation does not establish voice similarity
or pronunciation quality.
Model loading and narration/encoding stages log progress to stderr; one-shot
generation initializes its model each time unless a voice worker is reused.

### Extract a script

Call `extract_script` with `media_path` set to an existing absolute audio/video
path inside the configured workspace or data root. Defaults are
`model: "small.en"`, `language: "en"` and `offline: true`.

Offline mode requires cached recognition assets. `offline: false` permits
model downloads; it does not enable cloud rewriting. The result contains a
transcript path, timing, and Script JSON in each `scripts[*].content`, alongside
the saved script paths. Recordings may produce multiple parts. Review recognized
names and words against the source before generating new narration.

### Generate a reel edit guide

Call `generate_reel` with exactly one non-null source:

| Source | Arguments | Behavior |
|---|---|---|
| Approved script | `script`, actual `voice`, optional `seconds` | Narration plus edit guide; words unchanged |
| Existing recording | `audio_path` | Offline ASR captions/timing and original audio, no new voice |
| Text request | `prompt`, actual `voice`, optional `seconds` | Cloud writing, narration and edit guide |

Do not send placeholder empty strings for unused sources. On the existing-audio
branch, input must be at most 180 seconds; `voice` and `seconds` do not alter
the source. Captions need ASR review. The video adds a default 0.75-second closing
hold. On script/prompt input, the video has a +/-33% target-duration buffer,
including its closing hold; exact duration is not promised.

Output includes a 1080x1920 placeholder guide, audio, timeline, captions and
status paths. The user must replace numbered placeholders with their own clips.
Music is off by default; optional `music_path` must also be inside configured
roots. Do not invent a finished-footage or image-generation capability.

## Files, execution and handoff

Use absolute file paths on the server machine, not paths on an unrelated cloud
host. Relative MCP file paths resolve from `YAPIT_WORKSPACE`, not the repository
root. Put inputs under the configured workspace/data roots; paths outside them
are rejected.

Calls are synchronous and serialized per server. CPU generation may take
minutes; the client needs a suitable tool timeout. A client timeout is not proof
the underlying work stopped. Do not blindly retry generation after a timeout;
inspect known output/status files from a local terminal first. There is no
background queue or cross-process GPU lock.

Returned paths are local artifacts, not download URLs. Give the user the actual
returned paths, flag required listening/ASR/footage review, and distinguish an
app launch from enrollment completion. Default locations are
`data\generated_audio`, `data\generated_video`, `data\transcripts` and
`workspace\outputs`; configured roots may differ.

## References

- [Project README and installation](https://github.com/Arka95/yapit#readme)
- [MCP guide](https://github.com/Arka95/yapit/blob/main/docs/skills/README.md)
- [Enrollment guide](https://github.com/Arka95/yapit/blob/main/docs/enrollment/README.md)
- [Backend/model setup](https://github.com/Arka95/yapit/blob/main/docs/yapper/README.md)
