# YapIt developer guide

## Flat source layout

Source lives directly under `src` (for example `src\enrollment` and `src\rescript`);
there is no nested `src\yapit` folder. The public package/import name remains
`yapit`, and the CLI commands are unchanged.

```text
yapit\
  README.md
  VERSION
  pyproject.toml
  docs\
    dev.readme.md
    analysis.md
    audio-design.md
    module-architecture.md
    enrollment\
      README.md
      dev.readme.md
    rescript\
      README.md
      dev.readme.md
  src\
    __init__.py
    common_utils\
      __init__.py
      fileutils.py
      mediautils.py
      modelutils.py
    enrollment\
      __init__.py
      __main__.py
      cli.py
      desktop.py
      recorder.py
      session.py
      profiles.py
      artifacts.py
      sample_scripts.py
      sampleScripts\sample_<number>.json
      tests\
    rescript\
      __init__.py
      audio.py
      cli.py
      transcription.py
    cli.py
    server.py
    storage.py
    models.py
    constants.py
    media.py
    speech.py
    pipeline.py
    scripts.py
    timeline.py
    training.py
    voices.py
  scripts\
    Start Enrollment.cmd
    setup-mode.cmd
    yapit-mode.cmd
    run-mode.py
    setup-mode.py
    hatch-build.py
  tests\
  models\
    Qwen3-TTS-12Hz-1.7B-Base\
    cosyvoice3-2512\
    cosyvoice_source\
    hub\
  data\
    <profile>\audio\
    audio\
    transcripts\
    generated_audio\
  workspace\
    enrollment\<profile>\
    outputs\
  .venv\
  .venv-cpu\
  .venv-gpu\
  .venv-cosyvoice\
  .venv-cosyvoice-bootstrap\
```

Only `README.md` remains at the repository root; all project Markdown is in
`docs`. Model checkouts/cards and dependency documentation are third-party
assets, not project documentation to relocate. Runtime environments, models,
workspace contents and other profiles' data are Git-ignored. `data\arka` is
eligible for version control, including its audio; this does not stage or upload
it automatically. Package archives exclude all `data` contents. The CosyVoice runtime remains
separate and paused; this tree does not imply it is provisioned.

Hatch maps `src` to `yapit` in release wheels. `scripts\hatch-build.py` supplies
the corresponding editable-install import mapping without symlinks or another
source folder. The small `editables` dependency is declared for editable builds.
After moving source files, reinstall the package into each application runtime;
old editable installs still point at the removed folder. Do not install YapIt
into the separate Python 3.10 CosyVoice worker.

`VERSION` is the single source of the package version. Hatch reads its plain
version string for wheel, source and editable builds; `pyproject.toml` declares
the version as dynamic. Change `VERSION` for the next release, then build with
`.\.venv\Scripts\python.exe -m build --no-isolation`.

For the approved-feed workflow with build isolation disabled, install the
declared build dependencies first:

```powershell
.\.venv\Scripts\python.exe -m pip install "hatchling>=1.27,<2" "editables>=0.3,<1"
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

This guide covers development, component responsibilities, and advanced setup.
For the everyday enrollment -> script -> narration -> editing workflow, use
[README.md](../README.md). Audio research and the listening-validation checklist
are in [audio-design.md](audio-design.md).

## Local development

Use Python 3.11 or 3.12. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev,mcp]"
Copy-Item .env.example .env
.\.venv\Scripts\yapit doctor
```

Keep an existing `.env` rather than overwriting it. The editable install uses
the flat `src` directory through the public `yapit` namespace. `dev` installs pytest, Ruff, build tooling,
and MCP for protocol tests; `mcp` enables the stdio server. Neither requires
voice-model weights. Tkinter is supplied by a Tk-enabled Python installation,
not a new Python dependency.

Run commands from the repository root so relative workspace paths and local
configuration resolve consistently. Use absolute `YAPIT_WORKSPACE` and
`YAPIT_DATA_DIR` and `YAPIT_MODELS_DIR` when
launching from another directory or an MCP client.

Completed work in this checkout should be validated, committed, and pushed to the
configured remote as `Arka95`. Use the owner's local `arka95git` SSH key or an
unlocked SSH agent; never store its passphrase in documentation, scripts, Git
configuration, or commits. Stage only the task's changes, preserving unrelated
worktree edits. If authentication or the push fails, report that explicitly rather
than claiming the work was published.

## Configuration and dependencies

### Isolated CPU/GPU runtimes on Windows

`scripts\setup-mode.cmd cpu` and `scripts\setup-mode.cmd gpu` use normal isolated venvs,
preserving the old `.venv`. `scripts\setup-mode.py` installs the selected
`requirements-cpu.txt` / `requirements-gpu.txt` first without transitive dependency
resolution on the PyTorch index, then resolves dependencies from PyPI with the editable project
with `[voice,rescript,mcp]` under the same wheel constraints and
`requirements-runtime.txt`. Torch and torchaudio are both exactly 2.9.1+cpu
or 2.9.1+cu128, respectively. Runtime pins are transformers 4.57.3,
accelerate 1.12.0, faster-whisper 1.2.1, and CTranslate2 4.8.4.
Qwen 0.1.1 supplies gradio, librosa, soundfile, sox, onnxruntime, einops and
the remaining speech dependencies. Other transitive versions are resolved by
pip, not a complete lockfile. No global Python packages are installed.

Setup installs Hatchling/editables into the selected environment and uses them
for the editable build without another isolated build dependency download.
If package downloads fail, do not bypass TLS validation. For a trusted local
wheel directory, `.\scripts\setup-mode.cmd cpu --wheelhouse .\verified-wheels` (or `gpu`)
resolves runtime/build dependencies offline. PyTorch still uses its official
index/cache, and the SoX archive still comes from its checksum-verified mirror.
Only use public wheels whose upstream hashes or provenance you have verified.

### Dependency management on `DESKTOP-A05R1KA`

The owner reports that this Windows machine's VPN restricts PyPI access.
Direct wheel/metadata requests have produced TLS `HandshakeFailure`; a successful
index lookup is not proof that its redirected artifact host is reachable.
The configured organization feed is
`https://packagefeedproxy.microsoft.io/pypi/simple/`. pip successfully downloaded
the bootstrap uv wheel from that feed, while uv dependency resolution still
attempted blocked `files.pythonhosted.org` downloads. These are observations from
this machine, not a claim that either installer always works.

Prefer approved feeds and existing verified caches. If the feed lacks a wheel,
prepare a wheelhouse on an authorized Windows machine/network using the exact
backend requirements and matching interpreter/architecture. For the isolated
CosyVoice trial, clone its source first, then collect/install its dependencies:

```powershell
# Authorized matching Windows/Python 3.10 environment:
python -m pip download --dest .\verified-wheels -r .\requirements-cosyvoice.txt

# Restricted machine, after verifying wheel provenance and hashes:
.\.venv-cosyvoice-bootstrap\Scripts\python -m pip --python .\.venv-cosyvoice\Scripts\python.exe install --no-index --find-links .\verified-wheels -r .\requirements-cosyvoice.txt
```

Source distributions can require additional build tools/dependencies; a wheelhouse
is complete only after installation succeeds without network access in a fresh,
matching environment. Include the matching CUDA wheels, build requirements where
needed, and all transitive dependencies. Record resolved versions and trusted
SHA256 hashes. Model files, SoX archives, and managed Python downloads are separate
artifacts and need their own approved provisioning paths.

Do not disable certificate validation, change VPN policy, or substitute an
unreviewed public mirror. Where corporate TLS inspection requires a CA certificate,
use only the organization's supplied trust configuration. Keep index choices
local to a command/environment, never embed credentials in tracked files, and
do not modify the working Qwen `.venv-cpu`/`.venv-gpu` while resolving CosyVoice's
different requirements.

The initial CosyVoice source revision is
`074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`; its Matcha-TTS submodule is
`dd9105b34bf2be2230f4aa1e4769fb586a3c824e`. Model files are downloaded separately
to `models\cosyvoice3-2512`. `scripts\cosyvoice-trial.py` runs orchestration in the
normal YapIt environment, then starts a Python 3.10 worker for the incompatible
upstream dependencies. The worker explicitly checks CUDA placement of the
language model, flow decoder, and vocoder. The upstream Windows ONNX frontend may
remain CPU-only; reported PyTorch memory excludes other runtimes.

One verified alternative worked for the blocked NetworkX 3.1 wheel: download it
through the existing approved feed's artifact endpoint, verify its SHA256 against
the feed's published link, and install that local wheel with `--no-index --no-deps`
before continuing normal dependency resolution. Merely adding `--find-links`
did not prevent pip from selecting blocked remote metadata. This is a targeted
cache workaround, not proof that the entire CosyVoice environment is provisioned.

`scripts\cache-feed-wheel.py` implements that verified-cache path for individual
wheels on this machine. It accepts a wheel basename, requires a published SHA256,
uses only the approved feed artifact endpoint, and refuses modified cached files:

```powershell
.\.venv\Scripts\python.exe .\scripts\cache-feed-wheel.py networkx-3.1-py3-none-any.whl
.\.venv-cosyvoice-bootstrap\Scripts\python.exe -m pip --python .\.venv-cosyvoice\Scripts\python.exe install --no-index --no-deps .\.runtime-wheel-cache\cosyvoice\networkx-3.1-py3-none-any.whl
```

The wheel choices follow the official
[PyTorch previous-versions matrix](https://pytorch.org/get-started/previous-versions/).
The CUDA wheel supplies its own runtime and cuDNN. The installed NVIDIA
driver must support CUDA 12.8; the RTX 4070 SUPER supports this build.
Thea's `docs\setup.gpu.md` describes **OpenCV CUDA**, not Yapit/PyTorch:
do not reinstall its CUDA toolkit or cuDNN to configure these environments.
GPU wheels occupy several GB; the model cache remains shared and unchanged.

`scripts\run-mode.py`, wrapped by `scripts\yapit-mode.cmd`, sets child-only
`YAPIT_DEVICE`, `YAPIT_ASR_DEVICE`, and `YAPIT_ASR_COMPUTE_TYPE`.
CPU uses cpu/int8 for ASR; GPU uses cuda/float16. GPU mode validates CUDA
availability before dispatch and never falls back. ASR's explicit `--device`
and `--compute-type` override its mode defaults. An explicit ASR device alone selects the corresponding
default precision instead of retaining an incompatible mode precision.
`python <arguments>` runs
the selected interpreter; `mcp` runs its stdio server. Torch's environment-local
DLL directory is added to the GPU child's PATH for CTranslate2's CUDA 12/cuDNN 9
dependencies. No activation, process-global environment mutation, or execution
policy bypass is used.

```powershell
.\scripts\yapit-mode.cmd cpu python -m pip check
.\scripts\yapit-mode.cmd gpu python -c "import torch; x=torch.ones(2,device='cuda'); print(x+x)"
.\scripts\yapit-mode.cmd gpu python -m pip check
.\scripts\yapit-mode.cmd gpu mcp
```

To add developer tools, use the selected interpreter and retain constraints:
`.\scripts\yapit-mode.cmd cpu python -m pip install -c requirements-cpu.txt -c requirements-runtime.txt -e ".[dev]"`.
Use GPU constraints for GPU development. Neither setup downloads model weights;
`--offline` ASR reuses a previously cached model.

SoX setup downloads only the official 14.4.2 Windows archive, verifies SHA256
`8072cc147cf1a3b3713b8b97d6844bb9389e211ab9e1101e432193fad6ae6662`,
and installs only `sox.exe` and its DLLs into each environment's Scripts folder.
Existing different tool binaries are not overwritten. Source:
[SoX distribution](https://sourceforge.net/projects/sox/files/sox/14.4.2/).
The download is verified in memory; no user files or credentials are copied.

The supplied MCP config uses the mode launcher with absolute workspace/data roots.
Change its `gpu` argument to `cpu` when travelling. With direct MCP executables,
set device variables explicitly and ensure CUDA DLL discovery for GPU ASR.

CLI, desktop, and MCP entry points load `.env` through `python-dotenv`.
Existing process environment variables take precedence. `doctor` reports
optional package availability, the FFmpeg path, and CUDA availability when
PyTorch is installed; it does not download/load a speech model or prove that
inference works.

| Variable | Purpose / default |
|---|---|
| `YAPIT_WORKSPACE` | Generation metadata/edit guides and temporary enrollment files; defaults to `workspace` |
| `YAPIT_DATA_DIR` | Data/profile root; enrollment defaults independently to `.\data`; generation defaults beside the workspace |
| `YAPIT_MODELS_DIR` | Reusable model assets; defaults to `.\models`; shared SDK cache under `models\hub` |
| `YAPIT_MODEL` | Base Hub model ID or local model directory; defaults to `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| `YAPIT_DEVICE` | `auto` selects CUDA when available, otherwise CPU; can be set explicitly |
| `YAPIT_ASR_DEVICE` | Rescript device; defaults to `cpu` without a mode launcher |
| `YAPIT_ASR_COMPUTE_TYPE` | Rescript precision; mode launcher sets `int8` (CPU) or `float16` (GPU) |
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

**Reference-conditioned delivery (provisional Qwen baseline).** No head-to-head
model selection or expressive-quality benchmark has been measured. Environment
separation does not commit the project to this backend. Base cloning uses user recordings for
`lively-vlog`, `relaxed-travel`, and `asmr-travel`; it does not accept the native
emotion `instruct` supported by different CustomVoice/VoiceDesign model paths.
`voice-style-prompt` gives recording direction and an exact passage;
`voice-style-enroll` applies the normal consent, transcript, and 3-20 second
signal checks. Recordings live in `data\audio\styles\<voice>\<style>`;
`data\transcripts\styles\<voice>\<style>\reference.json` holds the sidecar,
and the voice profile maps the references. Default identity references remain
unchanged. Missing styles fail before model load; custom-voice checkpoints
cannot use this Base reference selection. There is no training, classifier,
or guaranteed emotion quality: real listening review remains mandatory.
The normal default can download weights on first use; installing the Python
packages alone is not an offline-ready setup. An explicitly selected voice
checkpoint takes precedence over `YAPIT_MODEL`.

## How the components work

**Entry points and contracts.** `cli.py` exposes Typer commands through `yapit`;
`server.py` exposes MCP tools through `yapit-mcp`. Both delegate generation to
`pipeline.py`, rather than maintaining separate rendering implementations.
`models.py` contains Pydantic contracts for briefs, scripts, voices, and
continuous timelines, rejecting unknown fields and invalid values.

**Native enrollment UI.** `enrollment\desktop.py` is a Windows Tkinter/ttk wizard with
playback through `winsound`. Blocking work runs in worker threads; results return
through a queue polled by Tk's event loop. `enrollment\recorder.py` discovers DirectShow
microphones and prepares a continuous mono 24 kHz PCM stream when a profile is opened.
Record saves that already-running stream to WAV immediately, including 250 ms of
in-memory pre-roll; unrecorded audio is otherwise discarded. Stop finalizes the
take while retaining the prepared stream. The WAV writer enforces the 20-second
audio limit. Finish/Close send FFmpeg `q` and release the microphone.
DirectShow uses a requested 50 ms buffer; device startup happens once, not per take.

**Enrollment state and curation.** `EnrollmentSession` saves each accepted take
against the exact displayed static passage, with resumable state in
`workspace\enrollment\<profile>\<profile>.session.json`, an inventory in
`data\<profile>\<profile>.journal.json`, and a manifest
in `<profile>.samples.json`. Five accepted samples make a profile usable; more can
be added later. There is no consent checkbox. Raw takes remain temporary until Finish;
optional preparation
trims outer silence and applies constant gain to a separate file.
`enrollment\profiles.py` checks duration, clipping, loudness, quiet frames, and duplicates,
preserves external source files and copies accepted mono 24 kHz samples into
`data\<profile>\audio`, estimates
speaking rate, and ranks one reference by signal characteristics. These checks
are not transcription, speaker verification, or proof of a noise-free sample.
Successful enrollment removes profile-owned workspace draft/raw/log/preparation
files, while retaining accepted audio and final metadata. Incomplete/failed
sessions remain resumable; data never holds enrollment logs.

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

**Shared utilities and model loading.** `common_utils\mediautils.py` owns FFmpeg
execution, input validation, audio-stream extraction/conversion, decoding,
resampling, WAV/MP3 encoding, image/video sampling and time formatting.
`common_utils\fileutils.py` owns streaming SHA-256 file hashing, atomic JSON writes, exclusive file copies,
safe lowercase naming and artifact-name reservations. Workflow policy remains
in `storage.py`, Rescript and enrollment rather than in the utility package.

`models.py` contains Pydantic data contracts; it is deliberately not an ML
loader. `common_utils\modelutils.py:load_model` takes a native SDK factory and
forwards its options with the centralized cache folder. Qwen uses
`cache_dir`, Whisper uses `download_root`, and a local-only factory can set
`cache_parameter=None`. Other compatible LLM SDK factories can use this same
entry point without changing profile/script schemas; this is not a claim that
every model shares one constructor or runtime. SDK imports and device/precision
selection stay lazy in their backends. Importing utilities or enrolling does
not import Torch/Whisper/Qwen or initialize model storage.

`constants.py` defines default model IDs and `models_directory()`, which creates
`YAPIT_MODELS_DIR` only when a model is requested. Existing local checkpoints
are loaded in place; reusable downloads use `models\hub`. The existing Whisper
small.en cache was adopted into that folder using hard links for immutable
large assets and copies for small metadata, leaving the original cache intact
and avoiding downloads/duplicate weight allocation. CPU and GPU modes share
the model folder. Do not remove external caches blindly.

**Orchestration and timing.** `pipeline.py` reserves a readable output name, saves flat
brief/provenance/status files under `workspace\outputs` and the script under `data\transcripts`,
synthesizes each beat, and inserts explicit
pauses. `common_utils\mediautils.py` fits audio using FFmpeg's pitch-preserving `atempo` only
within 0.90-1.10x. A larger adjustment is refused unless the unadjusted audio
already fits the +/-5 second tolerance. The encoded MP3 is decoded and checked
again before the run can complete. Failure writes `<name>.status.json` and `<name>.error.log`
for render jobs inside the pipeline's error-handling block.

**Media and edit guidance.** Shared media utilities handle decoding/resampling,
192 kb/s 48 kHz MP3 encoding with loudness processing and image/video extraction.
`media.py` retains the workflow-specific H.264/AAC placeholder guide renderer.
`timeline.py` scales measured beat lengths to the
encoded narration, splits longer beats, and prefers quiet windows near suggested
sub-cuts. Phrase captions are proportional word groups, not forced alignment.
The guide is numbered placeholder slates at 1080x1920 and 30 fps, not generated
travel footage or automatic editing of the source video.

**Storage.** `storage.py` owns workspace/data paths and confinement; common file
utilities provide naming, reservations, copying and atomic JSON replacement.
All new generated files are flat; `arka` is the
default generation voice. Existing `job` response fields identify the shared
metadata folder, not a private run folder; use `name`, `status_file`, and returned
artifact paths. MCP paths are resolved
and confined to the workspace or data root; CLI manifests may intentionally refer
to external local recordings, which are preserved in the canonical audio store.
Profile creation reuses lowercase names; imports merge and replace matching
filenames. `constants.py` defines the unified data filename template and default
profile; `common_utils.fileutils.data_filename` normalizes filenames to lowercase.
Enrollment has its own [README](enrollment/README.md),
[developer guide](enrollment/dev.readme.md), and local tests.
Generation has no durable job queue or
cross-process GPU lock, so do not run competing model-heavy processes on the
same constrained device.

## Data locations and supported boundaries

All modules share `DataStore` and `YAPIT_DATA_DIR` (default: the `data` directory
beside `YAPIT_WORKSPACE`). Set both roots to absolute paths for alternate working
directories, MCP clients, and WSL.

| Location beneath the data root | Content |
|---|---|
| `<profile>\audio\<profile>.audio_sample_<number>.wav` | Prepared static-passage recordings |
| `<profile>\audio\<profile>.<source>.*` | Accepted imported recordings and delivery references |
| `audio\imports` | Archived external inputs; video soundtrack extraction |
| `<profile>\<profile>.samples.json` / `.journal.json` | Exact sample manifests and inventory |
| `<profile>\<profile>.profile.json` / `.quality.json` | Usable voice profile and signal-quality reports |
| `transcripts\<profile>.<source>.script.json` / `.transcript.json` | Recognition scripts and sidecars |
| `transcripts\<name>.script.json` / `.captions.srt` | Rendering script and captions |
| `transcripts\<name>.train.jsonl` / `.holdout.jsonl` / `.train_codes.jsonl` | Training text and codec records |
| `generated_audio\<name>.wav` / `.mp3` / `.raw.wav` / `.fitted.wav` | Final audio and intermediates |

Profiles are stored only under `data\<profile>`. Enrollment receives a `DataStore`
directly; canonical profiles are never read from the workspace. Temporary
drafts, raw captures, capture logs, and preparation reports live in
`YAPIT_WORKSPACE\enrollment\<profile>` and are deleted after successful enrollment.
Under `YAPIT_WORKSPACE`, flat `outputs\<name>.*` files hold prompts/frames, timelines,
provenance, status, guides and logs. Model checkpoints retain the upstream-required
directory structure beneath the selected profile's `checkpoints\<readable-name>`.
Profiles and draft
state necessarily embed reference text as metadata; standalone transcript/script
exports use the transcript store. Accepted recordings and generated WAV/MP3
remain in data. An enrollment draft is not usable for
generation until **Finish enrollment** creates its voice profile.

There are no old profile/draft fallback paths. Application voice paths delegate
to the canonical data store. Rescript output
overrides must remain in their appropriate bucket; transcript `--output` must be
the transcript root, not a new child directory.
Externally supplied recordings are archived without modifying the source.
The selected `tests\data` golden fixture is the explicit versioned exception.

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
executable/workspace/data paths and explicit model/device configuration as needed;
do not depend on the client's working directory to find your `.env`.

The tools are `prepare_script_prompt`, `enroll_voice`, `generate_narration`,
`render_timeline`, `generate_reel`, and `get_job_status`; the reusable prompt is
`reel_script`. Generation accepts a Script object, not a script filename.
For example, `generate_reel` takes `voice: "my-voice"`, `seconds: 30`, and a
`script` object shaped like `examples\script.json`. Without a script or automatic
scripting, it returns a prompt awaiting approval.

All MCP file inputs, including recordings referenced inside sample manifests,
must be inside `YAPIT_WORKSPACE` or `YAPIT_DATA_DIR`. Relative paths still resolve
against the workspace; use absolute paths for the data root. Mutating/generation calls use a per-server lock
and reject concurrent work as busy. Calls are synchronous: configure a generous
client timeout and use one server per GPU. `get_job_status` accepts the returned
readable `name` through its compatibility argument `job_id`; legacy job directories
remain readable. It does not schedule background jobs.
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

Use the same workspace and data root; re-enroll if moving recordings between filesystems
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
checks, exact transcript pairing, workspace/data confinement, and filename-based
profile merging. Do not create duplicate profiles when adding recordings.
Add focused tests for changed behavior; do not weaken quality gates to make a
recording pass. For speech changes, also render with a consenting enrolled voice
and listen to held-out phrases before claiming a voice-quality improvement.
Documentation-only changes do not need an inference run.

Never commit model weights, generated media, `.env`, or credentials. Share private
recordings only with the owner's explicit approval.
`data\arka` and selected test fixtures such as `tests\data\arka.ladakhi_marmots.wav`
are exempt from the media-ignore policy; Arka's profile files are not automatically staged.
Review `git diff` and staged paths before
committing. Do not remove private workspaces or user recordings as test cleanup,
and do not add automatic uploads or checkpoint promotion.
