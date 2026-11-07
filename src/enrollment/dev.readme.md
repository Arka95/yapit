# Enrollment development

## Boundaries

`profiles.py` owns deterministic signal checks, minimum-sample validation,
filename-based merging, reference ranking, manifests, and the inventory journal.
`session.py` owns profile creation/resumption and exact passage-to-recording
pairing. `sample_scripts.py` loads numbered static JSON files in numeric order.
`artifacts.py` confines temporary files to the enrollment workspace and removes
profile-owned scratch files only after successful enrollment.
`recorder.py` controls Windows DirectShow recording through FFmpeg.
It requests a 50 ms device buffer and reads captured-audio timestamps from FFmpeg
progress in the workspace capture log. The UI announces readiness only after
positive captured audio, not a WAV header or elapsed launch time. FFmpeg enforces
the 20-second audio limit; startup time does not shorten a take.
`desktop.py` owns Tkinter UI and playback; blocking recording checks and writes
run off the Tk thread. `cli.py` and `__main__.py` expose the module independently.

The core accepts an explicit `DataStore`. Only application entry points load
`.env`. Shared contracts are in `models.py`, filename constants in `constants.py`,
and path/atomic JSON helpers in `storage.py`. Audio decoding and WAV writing use
the existing deterministic media helpers. There are no imports or calls to model
providers, script-writing APIs, transcription, inference, or training.

`yapit.voices` adapts application workflows to the enrollment helpers.
`yapit.desktop` remains a compatibility launcher. Main CLI and MCP enrollment
delegate to this module; the module CLI avoids importing unrelated pipelines.
Normal enrollment does not require a flag, and the UI has no consent control.

## Storage and updates

Profiles are normalized to lowercase. Shared bucket names (`audio`, `transcripts`,
`generated_audio`) cannot be profile names. All new files use
`storage.data_filename` and the template in `constants.py`.

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
real FFmpeg progress/graceful stopping with synthetic input, and actual Tk UI actions
without opening a microphone. These tests do not assess voice identity or model
quality. Static JSON baseline files must be present in the built wheel.
