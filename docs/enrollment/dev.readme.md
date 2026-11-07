# Enrollment development

## Boundaries

`profiles.py` owns deterministic signal checks, minimum-sample validation,
filename-based merging, reference ranking, manifests, and the inventory journal.
`styles.py` owns expressive-reference registration, named style guides and
profile/reference selection. It accepts a `DataStore`, preserves the baseline
reference and persists style recordings/metadata without loading a model.
`session.py` owns profile creation/resumption and exact passage-to-recording
pairing. `sample_scripts.py` loads numbered static JSON files in numeric order.
`artifacts.py` confines temporary files to the enrollment workspace and removes
profile-owned scratch files only after successful enrollment.
`recorder.py` controls Windows DirectShow recording through FFmpeg.
It requests a 50 ms device buffer and reads a continuous mono 24 kHz PCM stream
from FFmpeg. Profile/device selection prepares the stream in a worker thread;
readiness requires received PCM, not a WAV header. Record reuses the prepared
process and writes PCM to a raw WAV, including 250 ms of in-memory pre-roll.
The WAV writer caps each take at exactly 20 seconds while the stream stays ready
for the next take. Stop flushes the WAV; Finish and Close stop the FFmpeg process.
Unrecorded audio is discarded except for the small rolling memory buffer.
`desktop.py` owns Tkinter UI and playback; blocking recording checks and writes
run off the Tk thread. `cli.py` and `__main__.py` expose the module independently.

The core accepts an explicit `DataStore`. Only application entry points load
`.env`. Shared contracts are in `models.py`, filename constants in `constants.py`,
and path helpers in `storage.py`. Atomic JSON writes and file naming come from
`common_utils\fileutils.py`; decoding and WAV writing come from
`common_utils\mediautils.py`. There are no imports or calls to model
providers, script-writing APIs, transcription, inference, or training.

Main CLI and MCP enrollment delegate to this module; generation/training pass
their workspace's `DataStore` to its profile/style APIs. The old root
`voices.py` wrappers are removed rather than duplicated. The root `desktop.py`
compatibility shim is removed; `yapit enroll-app` still launches this module's
native `desktop.py`. The module CLI avoids importing unrelated pipelines.
Normal enrollment does not require a flag, and the UI has no consent control.
`yapit enroll-app`, `yapit voice-enroll` and `yapit profiles` are flat commands.
The standalone `python -m yapit.enrollment` CLI remains optional for isolated
module development.

## Storage and updates

Profiles are normalized to lowercase. Shared bucket names (`audio`, `transcripts`,
`generated_audio`, `generated_video`) cannot be profile names. All new files use
`common_utils.fileutils.data_filename` and the template in `constants.py`.

Prepared recording filenames use the static passage's stable number; importing
files uses their normalized basename. The journal is an inventory of accepted
recordings, not automatic speech verification. Raw microphone takes keep unique
IDs. Import merging validates the complete combined sample set before copying
files, then atomically replaces each JSON document. This is local single-user
storage, not a cross-process transactional store; do not edit a profile from
multiple processes simultaneously.

Raw WAVs, capture logs, draft snapshots, preparation reports, and curation staging
live in `workspace\enrollment\<profile>` (or `YAPIT_WORKSPACE`). Only accepted
audio, profile metadata, manifests, journals, and the aggregate quality report
remain in data. Clean audio publication uses a brief atomic replacement file,
also supporting separate workspace/data volumes. Failed/incomplete sessions keep
their scratch files. Successful UI/CLI enrollment deletes profile-owned scratch
files; external inputs and other profiles' sessions are untouched. Reopening a
completed profile rebuilds its slots from accepted audio and saved transcripts
without recreating completed draft files.

Opening an enrolled profile keeps existing recordings. When saved reading text
differs from imported profile text, the slot is not silently relabeled as matching
the static passage. Finishing exports a compatible `Voice`, retaining selected
checkpoint/speaker/style settings; no model weights are changed.

The shared `Voice` schema permits extra profile keys and preserves them on
serialization and enrollment merges. Known fields remain validated.
`writing_style` defaults to `energetic and informal`; optional `frequent_words`
and `preferred_greeting` default to `[]` and null. Missing metadata in older
profiles is supplied at load time and persisted on the next save, without a
bulk migration. Script/media contracts retain their existing extra-field policy.
Writing metadata is not yet consumed by Yapper's prompt builder or speech backend.

There are no legacy lookup or migration paths. `DataStore` owns canonical profile
paths and confines accepted recordings to the data root. The application
`Workspace` delegates profile paths to this store for generation/training; it is
not constructed by enrollment. UI and CLI factories use `data` or the
explicit `YAPIT_DATA_DIR`; `YAPIT_WORKSPACE` controls temporary files only.

## Tests and packaging

Tests live in `tests` inside this module and are included in repository pytest
discovery:

```powershell
.\.venv\Scripts\python.exe -m pytest src\enrollment\tests -q
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\ruff.exe format --check src tests
```

Coverage includes five-sample readiness, case-insensitive profile reuse,
filename replacement, retained unrelated audio/settings, inventories and hashes,
rejected duplicate/bad audio, rejected other-session recordings, successful
scratch cleanup, retained incomplete sessions/external inputs, clean data roots,
isolated no-model/no-network imports,
real FFmpeg prepared-stream reuse, WAV flushing and frame limits with synthetic input,
and actual Tk UI actions
without opening a microphone. These tests do not assess voice identity or model
quality. Static JSON baseline files must be present in the built wheel.
