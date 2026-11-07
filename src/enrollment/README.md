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
passage. Opening a profile prepares the selected microphone once. **Record**
enables when the microphone is ready, then starts saving immediately without
reopening it. A 250 ms in-memory pre-roll preserves speech beginning at the click.
The timer measures saved audio, not device startup time.
Replay the take, check **Confirm before saving**, and save. If the confirmation
is unchecked, **Save & Next** explains what is needed instead of staying grey.
There is no consent checkbox. Save **at least five distinct recordings**, then finish enrollment;
you can finish without recording the entire baseline and add more later.

The microphone stays active between takes; audio is not saved until **Record**.
**Stop** ends the take, not the microphone stream. **Finish enrollment** and
closing the window release the microphone. Changing the selected microphone
prepares the new device before enabling Record again.

The passage selector lets you revisit a saved passage. Saving a new take replaces
that passage's prepared audio at the same filename; raw takes remain temporary.
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

Only accepted audio and reusable profile metadata belong to `YAPIT_DATA_DIR`:

```text
data\arka\
  arka.profile.json
  arka.samples.json
  arka.journal.json
  arka.quality.json
  audio\
    arka.audio_sample_1.wav
workspace\enrollment\arka\
  arka.session.json
  arka.audio_sample_1.quality.json
  arka.audio_sample_1.raw.<take-id>.wav
  arka.audio_sample_1.raw.<take-id>.capture.log
```

`constants.py` defines the shared `<profile>.<name><suffix>` filename template,
the default profile, and the numbered audio-sample stem. `storage.data_filename`
enforces lowercase and sanitizes the stem. Recordings and metadata never go in
the source module; only the reusable baseline scripts do.

Raw takes are exactly what the microphone captured. Prepared samples are separate
copies with optional outer-silence trimming and constant volume gain; they are not
generated speech. Capture logs, raw takes, preparation reports, and draft state
stay in the temporary workspace. Successful **Finish enrollment** removes those
files, including unused/replaced takes. Save any wanted take before finishing.
Incomplete or failed sessions retain their workspace files for retry/resumption.
External import sources are never deleted. The data folder contains no capture logs.

`journal.json` (with the profile prefix shown above) lists accepted sample paths,
exact transcripts, file hashes, count, and the five-sample minimum. Draft state
retains its reading snapshot until completion. Reopening a completed profile
compares saved transcript text with the current baseline instead of silently
relabeling recordings. Signal checks reject clipping, excessive silence, unusable levels,
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

Profiles are read only from this data layout; there are no old-profile fallbacks.
Entry points use
`YAPIT_DATA_DIR`, or `data` beneath the current working directory independently
of `YAPIT_WORKSPACE`. Temporary enrollment files use
`YAPIT_WORKSPACE\enrollment\<profile>`, or `workspace` beside the data root.
The workspace must be outside the data root. Library callers pass an explicit `DataStore`.

See [dev.readme.md](dev.readme.md) for boundaries, tests, and implementation details.
