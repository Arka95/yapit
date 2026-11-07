# Rescript

Turn a local audio or video file into the same Script JSON that YapIt accepts
for narration: `title` and `beats` with `text`, `visual`, and `pause_ms`.
Rescript transcribes speech; it does not rewrite it or synthesize audio.

## Models and resource use

| Operation | Models | Default execution |
|---|---|---|
| `extract_audio` | None | CPU FFmpeg decoding/conversion and file copying |
| `transcript` / `transcribe-media` | Local Whisper ASR (`small.en`), with voice activity detection | CPU/int8 unless ASR settings or the GPU launcher select CUDA |
| Script composition | None | Deterministic sentence grouping, partitioning and schema validation |

ASR means automatic speech recognition. Whisper is a Transformer speech model,
not a separate chat/text-generation LLM. Rescript does not invoke Qwen TTS,
an LLM rewrite, or a cloud transcription service. Transcription is **not
model-free**: CPU/int8 can still consume substantial CPU time and RAM.
CUDA/float16 recognition can be GPU-intensive; GPU is optional, not required.
Larger `--model` choices can increase the workload. No training is performed.

Direct CLI use defaults to CPU/int8 when `YAPIT_ASR_DEVICE` and
`YAPIT_ASR_COMPUTE_TYPE` are unset; explicit options override those variables.
The GPU launcher also checks CUDA availability before dispatch, even for an
extraction command, but extraction itself does not run a neural model.

## Module boundaries and shared utilities

The implementation lives in `src\rescript`: `cli.py` owns command parsing and
JSON/error output, `audio.py` owns extraction/copy policy, and `transcription.py`
owns the recognizer adapter, timed segments, script composition and export workflow.
`__init__.py` exposes the two Python entry points. There is no desktop UI here.

Generic operations are shared rather than reimplemented:

- `common_utils\mediautils.py`: source validation, FFmpeg, audio-stream extraction and video detection.
- `common_utils\fileutils.py`: naming, exclusive copies, atomic JSON, reservations and streaming file hashes.
- `common_utils\modelutils.py`: lazy native model construction with the central cache directory.
- `storage.py`: data-bucket confinement and source archival, using the shared file helpers.

Whisper-specific settings and YapIt's script-partition rules remain in Rescript,
not in general-purpose utilities.

## Setup and use

From the YapIt checkout, in PowerShell:

```powershell
.\.venv\Scripts\python -m pip install -e ".[rescript]"
.\.venv\Scripts\yapit rescript extract_audio .\trip.mp4
.\.venv\Scripts\yapit rescript transcript .\data\audio\arka.trip.wav
```

Audio files work the same way, including WAV, MP3, M4A, and FLAC. FFmpeg
decodes the first audio stream of any supported local media file; a video
without audio fails. Inputs are limited to 250 MB; transcription also limits
decoded audio to one hour. Sources are not modified. URLs are not accepted.

### User-facing commands

`extract_audio INPUT` writes to the shared `data\audio` store.
Run from the repository root, or configure absolute `YAPIT_WORKSPACE` and
`YAPIT_DATA_DIR` values. `--output-dir` may select a subfolder inside the audio
store, but cannot redirect recordings outside that bucket.
Videos are converted to mono 24 kHz PCM WAV, matching YapIt's working audio.
Already supported audio containers (WAV, MP3, M4A, AAC, FLAC, OGG, Opus,
AIF/AIFF, WMA) are validated by decoding and then copied byte-for-byte.
Other decodable audio is converted to WAV. Content, not just the extension,
determines whether a file contains video; album-cover artwork is not a video.
Invalid/unsupported files, empty audio, and files without an audio stream
return an error. An existing destination file is never overwritten; an input
already stored at its canonical destination is validated and returned unchanged.
Extraction has a 250 MB input limit and a ten-minute processing timeout;
the one-hour duration limit applies to transcription, not extraction.

`transcript AUDIO` creates the reviewable transcript and Script JSON.
`transcribe-media INPUT` remains an equivalent command for direct audio/video
transcription. External audio inputs are archived under `data\audio\imports`;
direct video transcription also retains its extracted soundtrack there.

The default recognizer is faster-whisper `small.en`, running locally on CPU
with int8 computation. First use downloads model weights, not uploads of
your audio. Reusable model downloads use `models\hub` beneath `YAPIT_MODELS_DIR`.
For offline use, provision the model beforehand and pass
`--offline`; `--model` also accepts a local converted model directory.
GPU use is opt-in with `--device cuda --compute-type float16` and requires
the appropriate CTranslate2/CUDA libraries.

With YapIt's dual runtime setup, use `.\scripts\yapit-mode.cmd cpu rescript transcript`
or `.\scripts\yapit-mode.cmd gpu rescript transcript` followed by your input path.
The launcher sets CPU/int8 or CUDA/float16 for ASR and shares the same recordings,
transcripts, metadata, and cached models. GPU mode fails if the eGPU is absent;
CPU mode works without it. Explicit ASR device/precision options still override
these defaults. See the root README for provisioning both environments.

```powershell
.\.venv\Scripts\yapit rescript transcript .\speech.wav --output .\data\transcripts
.\.venv\Scripts\yapit rescript transcript .\speech.wav --model .\models\whisper-small-en --offline
```

Outputs are flat under `data\transcripts`; no run-ID directory is created.
`--output`, if supplied, must identify that same transcript root, not a child.
Names use `arka.<source>` (an existing `arka.` prefix is reused), with numeric
suffixes on duplicates. All modules use `YAPIT_DATA_DIR`, which
defaults to `data` beside the configured workspace. Generated narration belongs
in `data\generated_audio`; job metadata and video guides remain in the workspace.
The module can also run as `python -m yapit.rescript.cli`.

## Review and regenerate

A recording with a spoken span of at most 60 seconds produces `<name>.script.json`
when it fits YapIt's schema. Longer recordings produce `<name>.script-001.json`,
`<name>.script-002.json`, etc. Every split part is **over 30 and at most 60 seconds**.
A short, unsplit recording is allowed to be 30 seconds or less.

Cuts use recognized sentence endings, not arbitrary 60-second slices.
Partitioning considers the entire recording to avoid an undersized tail.
Punctuation is only a heuristic for meaningful cuts: review the context.
If a valid partition cannot be found, the command fails without exporting
partial scripts. It never silently truncates recognized speech.

`<name>.transcript.json` stores recognized text, approximate source timestamps,
script paths, and suggested narration durations. Leading/trailing silence is
excluded from the spoken span; internal silence counts. It is a review sidecar,
**not** a narration script. The CLI returns `status: needs_review`.

Listen to the source and correct recognition errors in each script, particularly
names, numbers, and quiet speech. Visuals default to `Supporting B-roll`;
Rescript does not analyze the video's images. Pauses follow recognized gaps,
capped at YapIt's 1200 ms limit; each script ends with no extra pause.

Pass an individual reviewed script to YapIt, using the CLI's
`suggested_seconds` as a starting point:

```powershell
.\.venv\Scripts\yapit narrate --script .\data\transcripts\arka.speech.script.json --seconds 45
```

Use the actual duration reported for your file, not necessarily 45.
Recognition timing does not guarantee synthesized duration or preserve the
original delivery. YapIt's normal duration checks still apply. Sub-five-second
clips suggest the minimum supported narration target of five seconds.
YapIt's voice synthesizer currently uses English; the default recognition
language is English too. Changing `--language` and `--model` does not add
multilingual synthesis support.

## Privacy and limitations

Rescript has no cloud transcription path. Temporary decoded audio is deleted
after recognition, including on failure. Transcripts and timestamps may still
contain private information. Its shared `data\audio` and `data\transcripts`
buckets remain Git-ignored; the project allows `data\arka` separately, but
does not automatically stage or upload any recording.
Existing exports in the workspace are not moved or deleted. The explicitly
selected `tests\data` audio/transcript pair remains a versionable fixture.
Recognition can miss speech or hallucinate words; silence detection is not
speaker identification. Multiple speakers are flattened into one narration
script without labels. Music, overlapping voices, and noisy recordings may
need manual correction.

See [dev.readme.md](dev.readme.md) for implementation and testing.

Model references: [Whisper](https://github.com/openai/whisper) and
[faster-whisper](https://github.com/SYSTRAN/faster-whisper).
