# Rescript developer guide

Rescript is an optional subpackage under `src\rescript`, shipped in
YapIt's wheel. Its speech-recognition dependency is isolated in the
`rescript` extra and imported only at recognition time. Existing narration,
enrollment, and MCP commands do not load Whisper or download weights.

## Development

From the repository root:

```powershell
.\.venv\Scripts\python -m pip install -e ".[dev,rescript]"
.\.venv\Scripts\python -m pytest tests\test_rescript.py
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m ruff format --check .
```

Unit tests use an injected recognizer or a fake Whisper module; they do not
download models or require a GPU. FFmpeg integration tests exercise real audio
decoding and video audio-stream extraction. For a listening smoke test, run
the public CLI on a known local recording, review `transcript.json` against
the source, and load each output with `yapit.scripts.load_script`. Validate
recognition quality separately from deterministic partitioning.

## Components and contracts

`cli.py` registers `yapit rescript extract_audio`, `transcript`, the equivalent
`transcribe-media` alias, and the
standalone module entry point. It loads `.env`, accepts local paths and explicit
recognition settings, and prints result JSON or an actionable error.

Optional `device`/`compute_type` arguments take precedence over
`YAPIT_ASR_DEVICE`/`YAPIT_ASR_COMPUTE_TYPE`. Without these ASR variables,
the default remains CPU/int8 regardless of the narration-only `YAPIT_DEVICE`.
An explicit device without explicit precision derives its compatible default
(CUDA/float16 or CPU/int8), rather than retaining the other mode's precision.
The dual mode launcher sets ASR variables and supplies the environment-local
Torch CUDA DLL directory on PATH for GPU recognition; it validates GPU
availability and never silently falls back.

`audio.py` uses `common_utils\mediautils.py` for source validation and decoding
the first audio stream with FFmpeg.
Its stream report distinguishes video from audio, excluding attached cover
art. Known audio containers are copied unchanged; video or other decodable
containers become mono 24 kHz WAV. Exclusive destination creation prevents
overwrites through `common_utils\fileutils.py`; failed copies remove only the
newly created partial file.
The shared `DataStore` selects `data\audio`; output overrides must stay in that
bucket. An already canonical audio input is returned unchanged after validation.
The directory is Git-ignored. Extraction is independent
of Whisper and does not download weights.

`transcription.py` contains the `TranscriptionBackend` protocol,
`WhisperTranscriber`, timed `Segment` contract, `compose_scripts`, and
`transcribe`. The Python entry point accepts an injected backend for testing.

Processing order:

1. Validate source and destination; decode the first audio stream to temporary
   mono 16 kHz PCM WAV with `common_utils.mediautils.extract_audio_stream`.
2. Reject empty or over-one-hour decoded audio before recognition. The decode
   limit includes a one-second sentinel, so overlong media is rejected rather
   than accepted after truncation.
3. Preserve external audio in `data\audio\imports` (or extract the soundtrack
   for direct video inputs). Existing canonical audio/generated-audio inputs
   are reused without another raw copy. Valid inputs are retained even when
   later recognition fails.
4. Run faster-whisper with VAD, word timestamps, no translation, and no
   previous-text conditioning. Consume its lazy generator inside stderr
   redirection so stdout remains CLI JSON.
5. Group words at sentence punctuation or 35 words into narration beats.
   Only sentence endings are eligible script boundaries. Validate finite,
   positive, ordered timestamps and their bounds against decoded duration.
6. Use dynamic programming to find globally viable groups. Minimize the
   number of parts, then prefer durations near 45 seconds. All parts of a
   split must be strictly over 30 and at most 60 seconds. Enforce the existing
   `Script` limits too: 40 beats, 650 words, and per-beat field constraints.
7. Reserve a readable `<profile>.<source>` filename after recognition and partitioning
   succeed. Write scripts and the sidecar directly in `data\transcripts` with
   atomic `common_utils.fileutils.save_json`; repeats use numeric suffixes, not UUID folders.
   Individual writes are atomic; the multi-file export is not transactional
   if the filesystem fails during export.

Source spans run from the first detected word to the last detected word.
Internal gaps are counted; at a cut, the following gap belongs to the next
part. Source timestamps remain only in `<name>.transcript.json` and the CLI result:
the strict YapIt `Script` schema receives no extra fields. Pauses are capped
and visuals are placeholders, so this is not a timing-preserving round trip.

Sentence punctuation is an intentional lightweight heuristic, not semantic
summarization or guaranteed topic separation. Unsplittable text, missing word
timestamps, invalid ordering, and no recognized speech fail explicitly.
Do not silently discard an incomplete final sentence or underlength tail.

## Dependencies and operations

The default is `small.en`, CPU/int8. CUDA/float16 and other models are explicit
CLI choices. `--offline` maps to Whisper's `local_files_only`; supply complete
local/cached weights for an offline run. The lazy shared `load_model` adapter
passes Whisper's `download_root` as `YAPIT_MODELS_DIR\hub`; model folder/defaults
are centralized in `constants.py`. No API keys or cloud permission
flags are used. `YAPIT_FFMPEG` retains its existing meaning.
`YAPIT_WORKSPACE` holds metadata/video/checkpoints, not new audio or transcript
exports. Metadata and guide outputs are flat. Model checkpoints and existing
voice/enrollment storage retain their required structure. `YAPIT_DATA_DIR` is the shared root for all modules, defaulting to `data`
beside the configured workspace. Its buckets are `audio`, `transcripts`, and
`generated_audio`. Audio overrides must remain in their bucket; transcript
overrides must be exactly the transcript root.
Tests isolate the data root with an environment override and verify output
placement as well as transcription behavior. Legacy artifacts remain readable;
no automatic media migration/deletion is performed.

There is no background queue, GPU lock, diarization, forced alignment, image
analysis, or MCP transcription endpoint. Avoid concurrent GPU-heavy runs
on a constrained shared machine. Temporary WAVs are removed by
`TemporaryDirectory`; exported transcript data remains until the user removes
it. Failures before export leave no transcript files or run directory.

Upstream API reference:
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), particularly
`WhisperModel`, word timestamps, VAD, model downloads, and CPU/GPU setup.
