# YapIt

Turn a script into narration in your own locally cloned voice, plus a portrait
reel-editing guide. Designed for English travel and honest product-review reels,
with target durations of 5-180 seconds.

**The output is audio and a 1080x1920 editing guide, not a finished reel.**
The video uses numbered dark placeholder frames; replace them with your footage
in your video editor.

## Setup

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

1. Choose a new voice alias, such as `my-voice`, select your microphone, confirm
   consent, and click **Open / resume alias**.
2. Read the displayed passage naturally: **Record -> Stop -> Play take**.
3. Confirm that you read the text exactly, then click **Save & Next**.
4. Repeat for all 12 passages, then click **Finish enrollment**.

Aim for 8-15 seconds per passage; recording stops at 20 seconds. Retake a passage
if you miss words or hear noise. Optional normalization and outer-silence trimming
preserve the original recording. Unusable signal levels, clipping, excessive
silence, and duplicate recordings are rejected.

Saved passages resume when you reopen the same unfinished alias. Drafts and takes
live under `workspace\enrollment\<alias>`; completed profiles live under
`workspace\voices\<alias>`. Existing enrolled aliases cannot be overwritten:
use a new name such as `my-voice-v2`.

The app is Windows-only and local-only. It needs no speech model or API key,
and accesses your microphone only after you press **Record**.

### Already have recordings?

Copy `examples\samples.example.json`, replace the audio paths and exact
transcripts, then enroll:

```powershell
.\.venv\Scripts\yapit voice-enroll my-voice --manifest .\my-samples.json --consent
```

Paths are relative to the manifest. Use complete 3-20 second clips with one
speaker, a quiet room, consistent microphone placement, and no music or clipping.
Listen to every clip and check its transcript yourself.

**Enrollment is reference-based voice cloning, not model training.** YapIt keeps
your samples and selects one reference for narration; it does not combine them
into an automatically trained voice. Signal checks cannot verify identity,
consent, background music, or transcript accuracy. Enroll only your own voice.

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

The command returns a `prompt`, its saved job directory, and any sampled image
paths. Paste the prompt into your assistant, attach the returned image/frames,
review the answer, and save the Script JSON as `script.json`.
`examples\script.json` shows the format: a title and beats containing `text`,
`visual`, and `pause_ms`.

For video, YapIt samples up to six frames roughly five seconds apart near the
start; it does not analyze the source audio. Supply verified facts for details
the images cannot establish. Use `--kind product` for product reviews.

A UTF-8 `.txt` script also works: sentences become beats with default pauses.
JSON gives you more control over pauses and visual suggestions. No API key is
needed for this prompt-and-review workflow.

## 3. Generate narration and the editing guide

Use the enrolled alias and the same target duration you used for the script:

```powershell
.\.venv\Scripts\yapit reel my-voice --script .\script.json --seconds 45
```

For audio and timeline only:

```powershell
.\.venv\Scripts\yapit narrate my-voice --script .\script.json --seconds 45
```

The JSON response gives the output paths. Each render gets its own directory
under `workspace\outputs\<job-id>`:

| File | Use |
|---|---|
| `narration.mp3` | Narration to import into your editor |
| `narration.wav` | Intermediate WAV narration before final MP3 loudness processing |
| `script.json` | The script used for this render |
| `timeline.json` | Measured timing, suggested cuts, and visual suggestions |
| `video\guide.mp4` | 9:16, 1080x1920, 30 fps placeholder video with narration |
| `video\captions.srt` | Approximate phrase captions for editing |
| `provenance.json` / `status.json` | Voice mode, synthetic-audio notice, and job result |

`narrate` omits the video and captions. Listen to the MP3 for voice similarity,
pronunciation, and pacing before publishing. Import it and the guide into your
editor, replace placeholder clips with your footage, and refine cuts/captions.

YapIt measures generated speech and pauses, then accepts a duration within
**+/-5 seconds**. Pitch-preserving tempo adjustment is limited to 0.90-1.10x;
larger mismatches fail rather than truncating speech. Shorten or expand the
script, or adjust `--seconds`, and rerun. Cut points and captions are editing
suggestions, **not word-level alignment**.

## Other ways to use YapIt

**MCP clients:** install `.[mcp]`, copy `examples\mcp.config.json` into your
client's MCP settings, and adjust its absolute paths. Tools can prepare prompts,
enroll voices, generate narration/reels, render timelines, and read job status.
Without a script, `generate_reel` returns a prompt first. MCP file inputs must
stay inside `YAPIT_WORKSPACE`.

**Optional automation and training:** automatic cloud script generation and
dedicated local fine-tuning are separate, opt-in features. Neither is required
for the workflow above. Setup and commands are in [dev.readme.md](dev.readme.md).

## Privacy and troubleshooting

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
