# Voice enrollment

Local, model-free profile and recording management. No LLM, ASR, TTS, GPU,
network request, API key, or model download is involved. This module does not
train weights or automatically verify spoken words.

## Recording UI

```powershell
.\.venv\Scripts\yapit enroll-app
# The isolated module entry point:
.\.venv\Scripts\python.exe -m yapit.enrollment enroll-app
```

Create or select a profile, choose the microphone, and record the displayed
passage. Replay the take, confirm the text is exact, and save. There is no consent
checkbox. Save **at least five distinct recordings**, then finish enrollment;
you can finish without recording the entire baseline and add more later.

The passage selector lets you revisit a saved passage. Saving a new take replaces
that passage's prepared audio at the same filename; raw takes remain separate.
Profile names are lowercase and case-insensitive. Opening `ARKA` opens `arka`,
not another profile.

## Static baseline

`sampleScripts\sample_1.json` through `sampleScripts\sample_16.json` are shared
across all profiles. Each contains a title and the exact text to read. They cover
conversational vlog narration, itinerary sequencing, directions, times and prices,
distances, place names, questions, contrast, quiet descriptions, practical advice,
travel gear, and warm closings. They are original static passages, not generated
by an LLM or customized to a profile.

Read in your usual accent at a natural pace. Aim for 8-15 seconds, with a
20-second recording limit. Use a consistent microphone, a quiet room, complete
words, and no background music. Avoid exaggerated emotion unless deliberately
collecting a separately labeled delivery style.

These passages provide a starter dataset, not a validated phonetic corpus or a
guarantee of trained voice quality. Five clips make a profile usable for reference
selection; downstream training remains separate and currently requires at least
10 clips and 60 seconds, with 5-15 clean minutes recommended by YapIt's guardrail.

## Data and filenames

All runtime files belong to this repository's data root (`YAPIT_DATA_DIR`):

```text
data\arka\
  arka.profile.json
  arka.session.json
  arka.samples.json
  arka.journal.json
  arka.quality.json
  arka.audio_sample_1.quality.json
  audio\
    arka.audio_sample_1.wav
    arka.audio_sample_1.raw.<take-id>.wav
    arka.audio_sample_1.raw.<take-id>.capture.log
```

`constants.py` defines the shared `<profile>.<name><suffix>` filename template,
the default profile, and the numbered audio-sample stem. `storage.data_filename`
enforces lowercase and sanitizes the stem. Recordings and metadata never go in
the source module; only the reusable baseline scripts do.

`journal.json` (with the profile prefix shown above) lists accepted sample paths,
exact transcripts, file hashes, count, and the five-sample minimum. Draft state
retains its reading snapshot so edited baseline text does not silently relabel
old recordings. Signal checks reject clipping, excessive silence, unusable levels,
short clips, and repeated recordings under different filenames.

## Import or merge recordings

```powershell
.\.venv\Scripts\yapit voice-enroll arka --manifest .\my-samples.json
.\.venv\Scripts\python.exe -m yapit.enrollment profiles
```

The manifest has a `samples` array of `{"audio": "path.wav", "text": "Exact words."}`.
Paths are relative to the manifest or absolute. Files become
`data\<profile>\audio\<profile>.<source-stem>.wav`. Existing profiles merge by
normalized filename: new filenames add samples; matching filenames overwrite
audio and update its transcript. The merged profile must contain at least five
distinct recordings. Invalid inputs are checked before imported files replace
existing ones. Original external input files are not changed.

Enrollment reads and writes only this data layout. It neither initializes a
workspace nor searches old workspace/profile directories. Entry points use
`YAPIT_DATA_DIR`, or `data` beneath the current working directory independently
of `YAPIT_WORKSPACE`. Library callers pass an explicit `DataStore`.

See [dev.readme.md](dev.readme.md) for boundaries, tests, and implementation details.
