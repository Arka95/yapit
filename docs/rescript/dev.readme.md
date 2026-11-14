# Rescript development

[Back to YapIt](../../README.md) | [Usage](README.md) |
[Shared test policy](../dev.readme.md#testing-and-diagnostics)

## Boundaries

`audio.py` owns extraction; `transcription.py` owns recognition and script
composition; `cli.py` adapts them to flat commands. Public Python APIs are
`extract_audio` and `transcribe`. The optional isolated CLI is
`python -m yapit.rescript.cli extract_audio|transcript`.

Media decoding/resampling, hashing and publication primitives come from
`common_utils`; `DataStore` owns buckets. Sentence partitioning and Script
constraints remain Rescript policy, not generic utility code.

Extraction distinguishes real video from cover artwork, validates by decoding,
copies supported audio unchanged and converts other inputs to mono 24 kHz WAV.
Exclusive writes preserve existing files; handled failures remove only newly
owned partial output. Extraction has a ten-minute processing timeout and does
not import Whisper.

## Recognition and composition

1. Validate paths and decode the first audio stream to temporary mono 16 kHz WAV.
   A one-second sentinel rejects over-one-hour input without accepting a truncation.
2. Archive external input, or extract a video soundtrack. Existing canonical
   audio is reused; retained sources survive later recognition failure.
3. Run lazy faster-whisper with VAD/word timestamps, no translation and no
   previous-text conditioning. Consume lazy SDK output under stderr redirection.
4. Validate finite, positive, ordered timestamps against the decoded duration.
   Group words at punctuation or 35 words; only sentence endings can split parts.
5. Find globally valid parts with dynamic programming: fewest parts, then near
   45 seconds. Split parts must be over 30 and at most 60 seconds and satisfy
   Script limits (40 beats/650 words).
6. Reserve output names after successful composition, then atomically write each
   script and sidecar. Multi-file export is not a filesystem transaction.

Spoken span runs from the first to last detected word; internal gaps count.
At a split, the following gap belongs to the next part. Source timestamps stay
in the sidecar, not the strict Script schema. Temporary WAVs are removed on
success/failure. This is not diarization, forced alignment or a timing-preserving
round trip.

## Runtime and tests

Explicit recognition options override `YAPIT_ASR_DEVICE` /
`YAPIT_ASR_COMPUTE_TYPE`; without them library defaults remain CPU/int8,
independent of narration's `YAPIT_DEVICE`. An explicit device derives compatible
default precision. The launcher supplies GPU DLL paths.

`WhisperTranscriber` constructs a model per recognition call using the shared
`models\hub` disk cache. Offline defaults to true; opting into downloads
does not upload source audio. `transcribe` accepts an injected
`TranscriptionBackend` for units. There is no MCP transcription tool or
cross-process GPU lock.

```powershell
.\.venv\Scripts\python.exe -m pytest src\rescript\tests -q
```

Units use fake recognition/SDKs. Real decoding, subprocess and media workflows
belong in opt-in `tests\integration`. Evaluate recognition accuracy separately
from deterministic timestamp validation and partitioning.
