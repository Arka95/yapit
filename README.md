# Yapit

Your script, your locally cloned voice, and a 1080x1920 reel editing guide.
English narration; travel and honest product-review prompts. No agent framework.
Target durations: 5-180 seconds.

## Install (PowerShell)

```powershell
cd C:\projects\yapit
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev,mcp]"
Copy-Item .env.example .env
.\.venv\Scripts\yapit doctor
```

FFmpeg is supplied by `imageio-ffmpeg`; a system installation is optional.
Install an appropriate PyTorch/torchaudio pair for your hardware using the
official PyTorch installer, then install voice inference:

```powershell
.\.venv\Scripts\python -m pip install -e ".[voice]"
```

Qwen model weights download on first inference. NVIDIA CUDA is recommended;
CPU inference uses float32 and can be very slow. No GPU performance or voice
similarity claim is made for your machine. Native Windows inference may need
the SoX executable required by Qwen's dependencies; Linux/WSL2 is the fallback
if upstream dependencies fail on Windows.

## 1. Enroll your voice

Record clean, complete 3-20 second clips: one speaker, quiet room, consistent
microphone, no music, no clipping. Use your natural Indian-English delivery.
Start with several expressive 8-12 second clips; record 5-15 clean minutes if
you intend to fine-tune. These lengths are Yapit recommendations, not model guarantees.
Manually check every transcript against the audio.

Copy `examples\samples.example.json`, replace the paths/transcripts, and run:

```powershell
.\.venv\Scripts\yapit voice-enroll arka --manifest .\my-samples.json --consent
```

Audio paths are relative to the manifest. Enrollment copies mono 24 kHz WAVs
into `workspace\voices\arka` and reports signal checks in `quality.json`.
It rejects duplicate audio, very quiet/clipped recordings and excessive silence.
**It cannot detect background music, transcript mistakes, identity or consent.**
Use only your own voice; `--consent` records your explicit attestation.

Enrollment is **reference-based cloning, not training**. One ranked reference
is used consistently for generation; extra clips become the fine-tuning dataset.
Listen to the selected reference and generated narration before publishing.

## 2. Get a script prompt (no API key)

```powershell
.\.venv\Scripts\yapit prompt --topic "A relaxed two-day trip" --seconds 30 --kind travel --facts "Use one base; keep the second afternoon free."
.\.venv\Scripts\yapit prompt --media .\trip.mp4 --seconds 30
```

The JSON response contains `prompt`, its saved location, and sampled image paths.
Paste the prompt into an assistant (including this one), attach the image/frames,
and save its JSON answer as `script.json`. The reusable base instructions live in
`src\yapit\script_prompt.txt`; `examples\script.json` shows the format.
For video, only six snapshots from the first 30 seconds are considered; no
source audio is analyzed. Supply facts for anything the frames cannot establish.

You can also provide a UTF-8 `.txt` script. Sentences become beats with default pauses;
JSON gives explicit `pause_ms` and per-beat visual suggestions.

## 3. Generate narration and the reel guide

```powershell
.\.venv\Scripts\yapit reel arka --script .\script.json --seconds 30
.\.venv\Scripts\yapit narrate arka --script .\script.json --seconds 30
```

`reel` produces `narration.mp3`, `narration.wav`, `script.json`, `timeline.json`,
`video\guide.mp4`, `video\captions.srt`, provenance and status.
`narrate` omits video rendering. The guide contains dark placeholder slates
numbered at suggested cut points, not generated footage. Each timeline cue
contains a visual suggestion.

Timing comes from synthesized beat audio plus explicit pauses. Long beats use
approximate sub-cuts; captions are beat/phrase guides, **not word-level alignment**.
The renderer accepts duration within **+/-5 seconds** and only applies
pitch-preserving tempo correction within 0.90-1.10x. Larger mismatches fail
with a rewrite instruction rather than truncating speech or stretching it badly.
If `reel` has a topic/media but no script, it returns a prompt awaiting your script.

Optional automated scripts use that same prompt, not an agent:

```powershell
.\.venv\Scripts\python -m pip install -e ".[cloud]"
$env:OPENAI_API_KEY = "<your-key>"
$env:YAPIT_ALLOW_CLOUD = "true"
.\.venv\Scripts\yapit reel arka --topic "My camera review" --kind product --facts "Only include these verified observations..." --seconds 30 --auto-script
```

Only the brief and image/video snapshots go to OpenAI, never voice recordings.
`YAPIT_SCRIPT_MODEL` selects the model. Credentials belong in environment variables
or ignored `.env`, never in scripts or Git.

## 4. Train a dedicated voice (optional; Linux/WSL2 + NVIDIA)

Use the **same workspace**, re-enroll under Linux if moving between filesystems,
install a CUDA-compatible PyTorch/torchaudio pair, then `.[train]` and a compatible
FlashAttention build. The official training script hardcodes FlashAttention/bfloat16;
Yapit does not silently patch it. Review and pin the upstream checkout:

```powershell
git clone https://github.com/QwenLM/Qwen3-TTS.git Qwen3-TTS
```

Run these commands in your Linux/WSL2 Python environment (path syntax there is Linux):

```text
yapit voice-train arka --upstream <reviewed-Qwen3-TTS-directory> --epochs 3 --batch-size 1
yapit voice-use-checkpoint arka --checkpoint <selected-checkpoint-directory>
```

Training requires >=10 distinct clips and >=60 seconds (minimum project guardrails,
not sufficient-quality guarantees). It holds out ~10%, prepares codec tokens, runs
official single-speaker fine-tuning, and records the upstream commit/base snapshot.
Checkpoints are **not auto-promoted**: compare held-out phrases and choose the
best epoch by listening, not merely the last epoch. VRAM needs are workload-dependent;
batch size 1 is not a promise that a small GPU can train. Existing reference cloning
continues until you select a checkpoint. Only reviewed local checkpoints are supported.

## MCP

Copy `examples\mcp.config.json` into your client's MCP settings, adjusting absolute
paths. It uses stdio, not an unauthenticated network listener.

Tools: `prepare_script_prompt`, `enroll_voice`, `generate_narration`,
`render_timeline`, `generate_reel`, `get_job_status`. Prompt: `reel_script`.
For example, call `generate_reel` with `voice: "arka"`, `seconds: 30`, and a
`script` object matching the example. With no script, it returns the prompt first.
All MCP input files, including audio paths inside sample manifests, must be
inside `YAPIT_WORKSPACE`. Use one MCP server per GPU and a generous client tool
timeout for synthesis; calls are synchronous, not background jobs.

## Development

```powershell
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m ruff format --check .
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m build
```

Tests use deterministic fake speech (not a fake clone) with **real MP3/MP4 encoding**
and MCP protocol checks; no API key/GPU/model download. CI covers Windows and Linux.
See `docs\audio-design.md` for the dated research, architecture, tradeoffs and
listening-validation checklist. Git ignores recordings, generated media, weights
and credentials. This project has no remote, automatic commits, or uploads.
