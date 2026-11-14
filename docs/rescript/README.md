# Rescript

[Back to YapIt](../../README.md) | [Developer guide](dev.readme.md)

Extract local audio and transcribe it into reviewable YapIt Script JSON.
Extraction uses FFmpeg without a model; transcription runs local Whisper
(`small.en` by default), not hosted ASR or a script-rewriting LLM.

## Use

The standard setup includes recognition dependencies:

```powershell
.\yapit.cmd extract_audio .\trip.mp4
.\yapit.cmd extract-script .\speech.wav
# Permit model downloads when the recognition cache is empty:
.\yapit.cmd extract-script .\speech.wav --no-offline
```

Transcription defaults to **offline** cached/local models. `--model` accepts a
model name or converted local directory. Your saved runtime selects CPU/int8 or
CUDA/float16; Python `transcribe` also accepts explicit device/precision overrides.

`extract-script` / MCP `extract_script` use the same recognition workflow and
also return the actual Script JSON in `scripts[*].content`. Python exposes
`yapit.rescript.extract_script`.

Local audio/video files are accepted, not URLs. Inputs are limited to 250 MB;
transcription additionally limits decoded audio to one hour. Videos need an
audio stream. Supported audio containers are validated and copied unchanged;
video/other decodable containers become mono 24 kHz WAV. Sources are preserved,
and existing output files are not overwritten.

## Review the exports

Audio lands under `data\audio`; external transcription sources are archived in
`data\audio\imports`. Scripts/sidecars go directly under `data\transcripts`,
with numeric suffixes for duplicate names.

| Export | Purpose |
|---|---|
| `<name>.script.json` | Narration script for a spoken span up to 60 seconds |
| `<name>.script-001.json`, etc. | Longer recording split into parts, each over 30 and at most 60 seconds |
| `<name>.transcript.json` | Recognized text, approximate source timestamps, script paths and suggested durations |

The result is `needs_review`, not approved narration. Correct words, names and
numbers against the source before generating:

```powershell
.\yapit.cmd generate-audio --voice myvoice --script .\data\transcripts\testuser.speech.script.json --seconds 45
```

Use the actual returned script path and suggested duration; 45 is only an example.
The sidecar is not a narration script. Source timing does not guarantee
synthesized duration. Cuts use sentence-ending heuristics; an unsplittable
recording fails rather than exporting truncated/partial scripts.

Visuals are placeholders, pauses follow recognized gaps (capped at 1200 ms),
and parts end without an extra pause. Recognition has no speaker labels or video
image understanding and can hallucinate/miss words. Changing recognition
language does not add multilingual synthesis.

Audio extraction's `--output-dir` must stay inside the audio store. Scripts are
always written to the transcript root, not a child folder. Configuration is in the
[shared storage guide](../dev.readme.md#storage-and-configuration).
