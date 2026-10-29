# YapIt developer guide

This guide covers development, component responsibilities, and advanced setup.
For the everyday enrollment -> script -> narration -> editing workflow, use
[README.md](README.md). Audio research and the listening-validation checklist
are in `docs\audio-design.md`.

## Local development

Use Python 3.11 or 3.12. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev,mcp]"
Copy-Item .env.example .env
.\.venv\Scripts\yapit doctor
```

Keep an existing `.env` rather than overwriting it. The editable install uses
the `src\yapit` package directly. `dev` installs pytest, Ruff, build tooling,
and MCP for protocol tests; `mcp` enables the stdio server. Neither requires
voice-model weights. Tkinter is supplied by a Tk-enabled Python installation,
not a new Python dependency.

Run commands from the repository root so relative workspace paths and local
configuration resolve consistently. Use an absolute `YAPIT_WORKSPACE` when
launching from another directory or an MCP client.

## Configuration and dependencies

CLI, desktop, and MCP entry points load `.env` through `python-dotenv`.
Existing process environment variables take precedence. `doctor` reports
optional package availability, the FFmpeg path, and CUDA availability when
PyTorch is installed; it does not download/load a speech model or prove that
inference works.

| Variable | Purpose / default |
|---|---|
| `YAPIT_WORKSPACE` | Private data and jobs; defaults to `workspace` |
| `YAPIT_MODEL` | Base Hub model ID or local model directory; defaults to `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| `YAPIT_DEVICE` | `auto` selects CUDA when available, otherwise CPU; can be set explicitly |
| `YAPIT_FFMPEG` | Optional FFmpeg executable override; otherwise uses `imageio-ffmpeg` |
| `YAPIT_ALLOW_CLOUD` | Must be `true` for automatic cloud scripts; otherwise disabled |
| `OPENAI_API_KEY` | Needed only for automatic cloud scripts |
| `YAPIT_SCRIPT_MODEL` | Optional script model override; defaults to `gpt-4.1-mini` |

**Speech dependencies.** Install a compatible PyTorch/torchaudio pair for the
target machine, then `.\.venv\Scripts\python -m pip install -e ".[voice]"`.
The voice extra uses `qwen-tts==0.1.1`. CPU inference uses float32; CUDA uses
bfloat16 when supported, otherwise float16. Inference uses SDPA attention,
not the training path's FlashAttention requirement.

**Windows tools.** FFmpeg handles capture and media conversion; the bundled
`imageio-ffmpeg` executable is the default. Qwen's dependencies may also need
SoX and its accompanying DLLs. Put a verified SoX installation on `PATH`, or
place its executable/dependencies in `.venv\Scripts`: `speech.py` adds the
interpreter directory to `PATH` when it finds SoX there and none is already
discoverable. Do not disable TLS, integrity checks, or repository trust controls
to work around package-installation failures.

**Offline inference.** First provision the complete base model and speech
tokenizer, then set `YAPIT_MODEL` to their local model directory.
`HF_HUB_OFFLINE=1` prevents Hub network access for cached/local inference.
The normal default can download weights on first use; installing the Python
packages alone is not an offline-ready setup. An explicitly selected voice
checkpoint takes precedence over `YAPIT_MODEL`.

## How the components work

**Entry points and contracts.** `cli.py` exposes Typer commands through `yapit`;
`server.py` exposes MCP tools through `yapit-mcp`. Both delegate generation to
`pipeline.py`, rather than maintaining separate rendering implementations.
`models.py` contains Pydantic contracts for briefs, scripts, voices, and
continuous timelines, rejecting unknown fields and invalid values.

**Native enrollment UI.** `desktop.py` is a Windows Tkinter/ttk wizard with
playback through `winsound`. Blocking work runs in worker threads; results return
through a queue polled by Tk's event loop. `enrollment.py` discovers DirectShow
microphones and captures mono 24 kHz WAVs through an FFmpeg subprocess. Record
opens the microphone; Stop sends FFmpeg `q` so it can finalize the file.
Capture has a 20-second limit and is never started just by opening the app.

**Enrollment state and curation.** `EnrollmentSession` saves each accepted take
against the exact displayed passage, with resumable state in `session.json`
and a manifest in `samples.json`. Raw takes remain intact; optional preparation
trims outer silence and applies constant gain to a separate file.
`voices.py` checks duration, clipping, loudness, quiet frames, and duplicates,
copies accepted mono 24 kHz samples into the voice profile directory, estimates
speaking rate, and ranks one reference by signal characteristics. These checks
are not transcription, speaker verification, or proof of a noise-free sample.

**Script preparation.** `scripts.py` combines `script_prompt.txt`, a validated
brief, a word budget, and the Script JSON schema. The default flow returns this
prompt for a human-reviewed conversation; no agent framework or API call is
involved. UTF-8 text scripts are split into sentence/paragraph beats, while JSON
supports explicit pauses and visual suggestions. Optional cloud generation uses
the same prompt and permits one duration-estimate revision.

**Local voice synthesis.** `speech.py` lazily loads Qwen3-TTS Base 1.7B and builds
a clone prompt from one reference recording and its verified transcript.
`QwenSpeech` reuses that model/prompt across beats within a render; it does not
pool all enrolled recordings or update model weights. A selected fine-tuned
checkpoint switches generation to its named custom speaker. The `SpeechBackend`
protocol lets tests inject deterministic speech without loading Qwen.

**Orchestration and timing.** `pipeline.py` creates a job, saves the brief,
script, provenance, and status, synthesizes each beat, and inserts explicit
pauses. `media.py` joins audio and uses FFmpeg's pitch-preserving `atempo` only
within 0.90-1.10x. A larger adjustment is refused unless the unadjusted audio
already fits the +/-5 second tolerance. The encoded MP3 is decoded and checked
again before the job can complete. Failure writes `status.json` and `error.log`
for render jobs inside the pipeline's error-handling block.

**Media and edit guidance.** `media.py` handles decoding/resampling, 192 kb/s
48 kHz MP3 encoding with loudness processing, image/video frame extraction,
and H.264/AAC guide rendering. `timeline.py` scales measured beat lengths to the
encoded narration, splits longer beats, and prefers quiet windows near suggested
sub-cuts. Phrase captions are proportional word groups, not forced alignment.
The guide is numbered placeholder slates at 1080x1920 and 30 fps, not generated
travel footage or automatic editing of the source video.

**Storage.** `storage.py` centralizes workspace paths, safe aliases, unique job
IDs, and atomic JSON replacement. MCP paths are resolved and confined to the
workspace; CLI manifests may intentionally refer to external local recordings.
Completed aliases are not overwritten. Generation has no durable job queue or
cross-process GPU lock, so do not run competing model-heavy processes on the
same constrained device.

## Data locations and supported boundaries

Under `YAPIT_WORKSPACE`, `enrollment\<alias>` holds draft state, original takes,
prepared samples, and per-take reports. `voices\<alias>` holds curated WAVs,
`profile.json`, and `quality.json`. `outputs\<job-id>` holds prompt/render/training
artifacts and available status files. An enrollment draft is not usable for
generation until **Finish enrollment** creates its voice profile.

Briefs currently support English, `travel` or `product`, and 5-180 seconds.
Scripts contain 1-40 beats, at most 650 words overall, and pauses of 0-1200 ms.
Image/video context accepts JPEG, PNG, WebP, MP4, MOV, MKV, or WebM, with a
250 MB input limit. Video context is up to six snapshots roughly five seconds
apart near the start; source audio is not transcribed or sent for scripting.

Signal gates require complete 3-20 second clips, no more than 0.1% clipped
samples, RMS at least -40 dBFS, and no more than 60% very quiet 20 ms frames.
Normalization does not bypass these gates or repair clipped speech. Twelve
starter passages improve variety but do not establish phonetic completeness,
voice similarity, or sufficient training quality.

## Optional cloud scripting

```powershell
.\.venv\Scripts\python -m pip install -e ".[cloud]"
$env:OPENAI_API_KEY = "<your-key>"
$env:YAPIT_ALLOW_CLOUD = "true"
.\.venv\Scripts\yapit reel my-voice --topic "My camera review" --kind product --facts "Only include these verified observations..." --seconds 30 --auto-script
```

This sends the brief and any sampled visual context to OpenAI, never the enrolled
voice recordings. `--script` and `--auto-script` are mutually exclusive.
Keep credentials in process environment variables or ignored `.env`, never in
source, examples, logs, or commits.

## MCP integration

Install `.[mcp]` and adapt `examples\mcp.config.json` in your client's settings.
It launches `yapit-mcp` over stdio, not a public HTTP listener. Set absolute
executable/workspace paths and explicit model/device configuration as needed;
do not depend on the client's working directory to find your `.env`.

The tools are `prepare_script_prompt`, `enroll_voice`, `generate_narration`,
`render_timeline`, `generate_reel`, and `get_job_status`; the reusable prompt is
`reel_script`. Generation accepts a Script object, not a script filename.
For example, `generate_reel` takes `voice: "my-voice"`, `seconds: 30`, and a
`script` object shaped like `examples\script.json`. Without a script or automatic
scripting, it returns a prompt awaiting approval.

All MCP file inputs, including recordings referenced inside sample manifests,
must be inside `YAPIT_WORKSPACE`. Mutating/generation calls use a per-server lock
and reject concurrent work as busy. Calls are synchronous: configure a generous
client timeout and use one server per GPU. `get_job_status` reads persisted status
by the output directory's basename; it does not schedule background jobs.
Keep model/library output on stderr so stdout remains valid MCP protocol data.

## Optional dedicated fine-tuning

`training.py` wraps the official upstream single-speaker training scripts;
enrollment alone never runs it. Training requires Linux/WSL2, an NVIDIA CUDA GPU
with bfloat16 support, `.[train]`, and a compatible FlashAttention build.
Use a clean, reviewed and pinned checkout of the official upstream repository:

```powershell
git clone https://github.com/QwenLM/Qwen3-TTS.git Qwen3-TTS
```

Run the following inside the Linux/WSL2 environment with its own installed
dependencies and valid local paths:

```text
yapit voice-train my-voice --upstream <reviewed-Qwen3-TTS-directory> --epochs 3 --batch-size 1
yapit voice-use-checkpoint my-voice --checkpoint <selected-checkpoint-directory>
```

Use the same workspace; re-enroll if moving recordings between filesystems
invalidates their paths. Project guardrails require at least ten distinct clips
and 60 seconds; 5-15 clean minutes is a recommendation, not a quality guarantee.
The wrapper deterministically holds out about 10%, prepares codec tokens, and
records the upstream commit and base Hub snapshot used for training. This path
resolves its own base Hub snapshot; `YAPIT_MODEL` is an inference setting.

Checkpoint selection is manual: compare held-out phrases and choose by listening,
not just the last epoch. Only reviewed local custom-voice checkpoints containing
the alias's speaker are accepted. Reference cloning remains active until you
select one. Batch size 1 is not a guarantee of fitting a particular GPU.

## Validation and contribution instructions

Run the existing checks from the repository root:

```powershell
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m ruff format --check .
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m build
```

CI runs these checks on Windows and Linux with Python 3.12. Tests inject
deterministic synthetic speech but exercise real MP3/MP4 encoding and MCP
protocol behavior, with no API key or speech-model download. Windows-only
enrollment tests cover real FFmpeg graceful stopping using synthetic audio and a
Tk wizard workflow with a fake recorder; they do not capture your microphone.
Passing these tests does not establish real-model pronunciation or voice quality.

Keep changes focused and preserve the common pipeline, schemas, duration
checks, explicit consent, workspace confinement, and existing-alias protection.
Add focused tests for changed behavior; do not weaken quality gates to make a
recording pass. For speech changes, also render with a consenting enrolled voice
and listen to held-out phrases before claiming a voice-quality improvement.
Documentation-only changes do not need an inference run.

Never commit recordings, model weights, generated media, `.env`, or credentials.
These are covered by `.gitignore`, but review `git diff` and staged paths before
committing. Do not remove private workspaces or user recordings as test cleanup,
and do not add automatic uploads or checkpoint promotion.
