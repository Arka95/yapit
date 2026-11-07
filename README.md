# YapIt

Turn a script into narration in your own locally cloned voice, plus a portrait
reel-editing guide. Designed for English travel and honest product-review reels,
with target durations of 5-180 seconds.

**The output is audio and a 1080x1920 editing guide, not a finished reel.**
The video uses numbered dark placeholder frames; replace them with your footage
in your video editor.

## Setup

### Windows: CPU for travel, NVIDIA GPU at home

Keep your existing `.venv` and `.env`. Provision two independent Python 3.11/3.12
environments from the repository root (no activation or PowerShell policy changes):

```powershell
.\setup-mode.cmd cpu
.\setup-mode.cmd gpu
.\yapit-mode.cmd cpu doctor
.\yapit-mode.cmd gpu doctor
.\yapit-mode.cmd gpu --help
```

`.venv-cpu` contains matching `torch`/`torchaudio` 2.9.1 CPU wheels;
`.venv-gpu` contains matching 2.9.1 CUDA 12.8 wheels. Both install the project,
Qwen speech, Rescript, MCP, and checksum-verified Windows SoX.
GPU mode explicitly selects CUDA and **fails if the eGPU is disconnected**.
CPU mode explicitly selects CPU, regardless of the shared `.env`. Switch per command:

```powershell
.\yapit-mode.cmd cpu rescript transcript .\data\audio\trip.wav --offline
.\yapit-mode.cmd gpu rescript transcript .\data\audio\trip.wav --offline
.\yapit-mode.cmd gpu narrate my-voice --script .\data\transcripts\my-script.json --seconds 45
```

Rescript uses CPU/int8 or GPU/float16 in the selected mode. Both modes share
`data\audio`, `data\transcripts`, `data\generated_audio`, workspace metadata,
and the normal Hugging Face model cache; setup does not download model weights.
The launcher changes only the child process environment and runs from the checkout.
No system CUDA/cuDNN reinstall is needed for PyTorch's bundled CUDA runtime.
`Start Enrollment.cmd` defaults to CPU; `.\Start Enrollment.cmd gpu` selects GPU.

The older single-environment commands below remain supported; substitute
`.\yapit-mode.cmd cpu` or `.\yapit-mode.cmd gpu` for `.\.venv\Scripts\yapit`.
See [dev.readme.md](dev.readme.md) for exact dependencies and MCP configuration.
The current responsibility map and proposed reusable-library boundaries are in
[docs\module-architecture.md](docs/module-architecture.md); this is a plan, not an
implemented package split.

### VPN-restricted Windows machine: `DESKTOP-A05R1KA`

On Arka's Windows eGPU machine, the VPN restricts direct PyPI access. Treat package
downloads as a network constraint, not a CUDA/GPU failure. An index can be reachable
while the wheel host it redirects to is blocked; observed failures include TLS
`HandshakeFailure`. Changing from pip to uv does not remove the VPN restriction.

Use the configured organization-approved package feed where available. During
setup, pip successfully fetched a package through
`https://packagefeedproxy.microsoft.io/pypi/simple/`; this does not guarantee that
every dependency or redirected download is available. Prefer per-command index
settings rather than changing global configuration:

```powershell
.\.venv\Scripts\python -m pip install --index-url https://packagefeedproxy.microsoft.io/pypi/simple/ -e .
```

If the feed cannot supply the required dependencies, prepare a verified Windows
wheelhouse on an authorized network/machine with matching Python and CPU/CUDA
versions, then install locally. Existing runtime setup supports
`.\setup-mode.cmd gpu --wheelhouse .\verified-wheels` and the equivalent CPU mode.
That option covers runtime/build dependencies; PyTorch and SoX may still need
their official downloads/cache. Model weights must also be provisioned separately.
See the developer guide for an offline dependency-install example.

Keep separate backend environments: CosyVoice's Python 3.10/pinned dependencies
must not replace the existing Qwen CPU/GPU environments. Never disable TLS
verification, use untrusted mirrors, or commit package-feed credentials to work
around the VPN.

Use Python 3.11 or 3.12. Run these PowerShell commands from your checkout
(the example location is `C:\projects\yapit`):

```powershell
cd C:\projects\yapit
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
Copy-Item .env.example .env
.\.venv\Scripts\yapit doctor
```

If you already have `.env`, keep it and skip `Copy-Item`. FFmpeg is supplied by
`imageio-ffmpeg`; a system installation is optional.

For narration, install a matching PyTorch/torchaudio pair suitable for your
hardware, then add the speech dependencies:

```powershell
.\.venv\Scripts\python -m pip install -e ".[voice]"
```

The Qwen speech model downloads on first inference unless you configure a local
model directory. NVIDIA CUDA is recommended; CPU inference works but can be
very slow. Windows inference may also require SoX. See [dev.readme.md](dev.readme.md)
for configuration, offline use, and dependency troubleshooting.

## 1. Enroll your voice

### Windows recording app

Double-click **`Start Enrollment.cmd`**, or run:

```powershell
.\.venv\Scripts\yapit enroll-app
```

1. Create or select a voice profile, such as `arka`, choose your microphone,
   and click **Create / open profile**. There is no consent checkbox.
2. Read the displayed passage naturally: **Record -> Stop -> Play take**.
3. Confirm that you read the text exactly, then click **Save & Next**.
4. Save at least **five distinct samples**, then click **Finish enrollment**.
   The 16 shared static passages are under `src\enrollment\sampleScripts`;
   recording the full baseline is recommended, not required.

Aim for 8-15 seconds per passage; recording stops at 20 seconds. Retake a passage
if you miss words or hear noise. Optional normalization and outer-silence trimming
preserve the original recording. Unusable signal levels, clipping, excessive
silence, and duplicate recordings are rejected.

Reopen the same profile to add recordings or select a saved passage to replace
its audio. Names are always lowercase; `ARKA` and `arka` are one profile.
Recordings live under `data\<profile>\audio`, with profile metadata, session
state, manifests and journals under `data\<profile>`. Files use the shared
`<profile>.<name>.<extension>` format defined in `src\constants.py`, for example
`arka.audio_sample_1.wav` and `arka.journal.json`. Imports merge by normalized
filename and overwrite matching files; they do not create duplicate profiles.
Enrollment uses only the data root and does not create or scan a workspace.

The app is Windows-only and local-only. It needs no speech model or API key,
and accesses your microphone only after you press **Record**.

### Already have recordings?

Copy `examples\samples.example.json`, replace the audio paths and exact
transcripts, then enroll:

```powershell
.\.venv\Scripts\yapit voice-enroll my-voice --manifest .\my-samples.json
```

The combined profile needs at least five distinct recordings. Paths are relative
to the manifest. Use complete 3-20 second clips with one
speaker, a quiet room, consistent microphone placement, and no music or clipping.
Listen to every clip and check its transcript yourself.

**Enrollment is reference-based voice cloning, not model training.** YapIt keeps
your samples and selects one reference for narration; it does not combine them
into an automatically trained voice. Signal checks cannot verify identity,
background music, or transcript accuracy.

Enrollment is a CPU-only, zero-model module with its own UI, CLI, static scripts,
and tests. See [the module README](src/enrollment/README.md) and
[developer README](src/enrollment/dev.readme.md). Standalone entry point:
`python -m yapit.enrollment enroll-app`.

### Expressive narration from your own delivery

Qwen is the current **provisional baseline**, not a proven best backend;
environment isolation is independent of the eventual model choice.
The default Base model cannot accept native emotion instructions like
CustomVoice/VoiceDesign models. Instead, record a real example of your intended
delivery; YapIt clones from that reference, not a synthetic preset presented as you:

```powershell
.\yapit-mode.cmd cpu voice-style-prompt --style lively-vlog
.\yapit-mode.cmd cpu voice-style-enroll my-voice --style lively-vlog --audio .\expressive.wav --text "EXACT WORDS YOU RECORDED" --consent
.\yapit-mode.cmd gpu narrate my-voice --script .\data\transcripts\my-script.json --seconds 45 --style lively-vlog
```

Read the supplied short passage in your natural lively-vlog, relaxed-travel,
or asmr-travel delivery. Use one clean, user-owned 3-20 second recording and
an exact transcript; listen before enrolling. Missing style references fail
before loading the speech model, and styles cannot be used with a selected
custom-voice checkpoint. Default references and voice identity are unchanged.
This is not training or emotion classification, and expressive quality is not
guaranteed: listen to generated audio and review pronunciation, delivery, and timing.

### Experimental CosyVoice3 comparison

The first instruction-controlled challenger is
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`, with a separate `.venv-cosyvoice`
Python 3.10 environment and official source under `models\cosyvoice_source`.
It does not replace the default Qwen backend or its CPU/GPU environments.
After provisioning the dependencies and local model files, run:

```powershell
.\.venv\Scripts\python.exe .\scripts\cosyvoice-trial.py --script .\data\transcripts\arka.cappadocia_vlog.script.json
```

The trial requests warm, engaged, lightly curious travel narration from the Arka
reference, uses the upstream instruction-plus-reference interface, and keeps
`speed=1.0`. It joins the approved words without inserting the ASR source pauses;
the model still determines punctuation pauses. Outputs use the normal flat
profile/source naming and duplicate suffixes. Provenance records the actual
instruction, and metrics record generation time and PyTorch peak GPU memory.
Emotion adherence and voice similarity remain **unreviewed until listening**.
This is an experimental runner, not the finished `add_emotion` feature.

## 2. Prepare and approve a script

Script writing can stay conversational: ask your assistant for a catchy,
non-cringey travel or product-review script with your target duration. To get
YapIt's reusable instructions and JSON format:

```powershell
.\.venv\Scripts\yapit prompt --topic "A relaxed seven-day Istanbul and Cappadocia itinerary" --seconds 45 --kind travel --facts "Four days in Istanbul; three in Cappadocia. Balloon flights depend on weather."
```

You can provide an image or video instead:

```powershell
.\.venv\Scripts\yapit prompt --media .\trip.mp4 --seconds 30
```

The command returns a `prompt`, its flat metadata paths, and any sampled image
paths. Paste the prompt into your assistant, attach the returned image/frames,
review the answer, and save the Script JSON under `data\transcripts`, for example
`data\transcripts\my-script.json`.
`examples\script.json` shows the format: a title and beats containing `text`,
`visual`, and `pause_ms`.

For video, YapIt samples up to six frames roughly five seconds apart near the
start; it does not analyze the source audio. Supply verified facts for details
the images cannot establish. Use `--kind product` for product reviews.

A UTF-8 `.txt` script also works: sentences become beats with default pauses.
JSON gives you more control over pauses and visual suggestions. No API key is
needed for this prompt-and-review workflow.

## 3. Generate narration and the editing guide

Generation defaults to the enrolled `arka` profile. You can still pass another
enrolled alias explicitly. Use the same target duration you used for the script:

```powershell
.\.venv\Scripts\yapit reel --script .\data\transcripts\my-script.json --seconds 45
```

For audio and timeline only:

```powershell
.\.venv\Scripts\yapit narrate --script .\data\transcripts\my-script.json --seconds 45
```

The JSON response gives the output paths and a readable `name`, not a job ID.
New generated artifacts are flat: no per-run audio, transcript, or metadata
subfolders. Filenames use `<profile>.<source-filename>` without the source extension.
Already-prefixed names reuse the profile prefix. Unsafe filename characters become
underscores. For script-only CLI generation the source is the script filename;
inline MCP scripts use their title. Repeated names gain `.1`, `.2`, etc. before the
extension (or artifact suffix), without overwriting previous results.
An existing related artifact can also reserve a name. For example:

| File | Use |
|---|---|
| `data\generated_audio\arka.trip.wav` / `arka.trip.mp3` | Final narration |
| `data\generated_audio\arka.trip.raw.wav` / `arka.trip.fitted.wav` | Synthesis and fitting intermediates |
| `data\transcripts\arka.trip.script.json` | The script used for this render |
| `workspace\outputs\arka.trip.timeline.json` | Measured timing and suggested cuts |
| `workspace\outputs\arka.trip.guide.mp4` | 9:16 placeholder video with narration |
| `data\transcripts\arka.trip.captions.srt` | Approximate phrase captions |
| `workspace\outputs\arka.trip.provenance.json` / `arka.trip.status.json` | Voice mode, synthetic notice, and result |

`narrate` omits the video and captions. Listen to the MP3 for voice similarity,
pronunciation, and pacing before publishing. Import it and the guide into your
editor, replace placeholder clips with your footage, and refine cuts/captions.

YapIt measures generated speech and pauses, then accepts a duration within
**+/-5 seconds**. Pitch-preserving tempo adjustment is limited to 0.90-1.10x;
larger mismatches fail rather than truncating speech. Shorten or expand the
script, or adjust `--seconds`, and rerun. Cut points and captions are editing
suggestions, **not word-level alignment**.

## Other ways to use YapIt

**Rescript:** turn local audio/video into narration-compatible Script JSON with
`yapit rescript transcript .\recording.wav`. Install `.[rescript]` first.
Use `yapit rescript extract_audio .\recording.mp4` to extract audio to
`data\audio` or copy an already supported audio file there.
Long recordings are split at sentence endings into parts over 30 and at most
60 seconds; review recognition before generating narration. See
[Rescript README](src/rescript/README.md).

**MCP clients:** install `.[mcp]`, copy `examples\mcp.config.json` into your
client's MCP settings, and adjust its absolute paths. Tools can prepare prompts,
enroll voices, generate narration/reels, render timelines, and read job status.
Without a script, `generate_reel` returns a prompt first. MCP file inputs must
stay inside `YAPIT_WORKSPACE` or `YAPIT_DATA_DIR`.

**Optional automation and training:** automatic cloud script generation and
dedicated local fine-tuning are separate, opt-in features. Neither is required
for the workflow above. Setup and commands are in [dev.readme.md](dev.readme.md).

## Privacy and troubleshooting

All modules use one data layout: **`data\audio`** for imported/raw recordings
(including curated enrollment copies), **`data\transcripts`** for scripts,
transcripts, training text, and captions, and **`data\generated_audio`** for
synthesized WAV/MP3 files and their audio intermediates. New generation and
Rescript artifacts have readable flat filenames, without job-ID folders.
Voice/enrollment storage retains its existing alias subfolders. Rescript stores
`arka.<source>.script.json` and `arka.<source>.transcript.json` directly in `data\transcripts`
and archives externally supplied audio under `data\audio\imports`.
Set `YAPIT_DATA_DIR` to choose a different data root; by default it is `data`
beside the configured workspace. Set absolute workspace/data paths when launching
from another directory. Job metadata and video guides remain in `workspace\outputs`;
voice profiles and enrollment recordings belong only to `data\<profile>`.
Enrollment independently defaults to `.\data`, ignores `YAPIT_WORKSPACE`, and has
no old-profile/draft lookup. The explicitly selected
audio/transcript fixture under `tests\data` is a versionable test-only exception.

Recordings and narration are processed locally. Model/dependency downloads need
network access; optional cloud script generation sends the brief and sampled
visuals, never enrollment recordings. Manually attaching media to an assistant
is a separate sharing decision. Private data, weights, generated media, and
`.env` are Git-ignored; YapIt does not automatically commit or upload them.

Run `.\.venv\Scripts\yapit doctor` to report dependency availability. If recording
is too quiet, check the selected microphone and input gain and move closer;
normalization cannot repair clipping or recover missing speech. For setup,
implementation details, tests, and contributor instructions, see
[dev.readme.md](dev.readme.md).
