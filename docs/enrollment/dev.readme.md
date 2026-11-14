# Enrollment development

[Back to YapIt](../../README.md) | [Usage](README.md) |
[Shared development policy](../dev.readme.md#testing-and-diagnostics)

## Ownership

| Component | Responsibility |
|---|---|
| `profiles.py` | Signal gates, minimum samples, filename merging, reference ranking and inventories |
| `styles.py` | Expressive references and voice/style selection |
| `session.py`, `sample_scripts.py` | Resumable passage/recording pairing and numbered static scripts |
| `artifacts.py` | Profile-owned scratch paths and successful-session cleanup |
| `recorder.py`, `desktop.py` | FFmpeg DirectShow lifecycle, Tk UI and playback |
| `launcher.py` | Independent app launch with configured roots and isolated streams |

Core APIs accept `DataStore`; entry points load `.env`. Shared contracts,
storage, file naming/hashing and media decoding stay in their existing owners.
Enrollment does not import speech SDKs, construct a generation Pipeline or run
ASR/training. `python -m yapit.enrollment` provides an isolated module CLI.

The public `enroll_voice()` action takes no inputs and opens the existing app.
CLI `enroll-voice` and MCP use this same launcher; `enroll-app` is not an alias.
It starts the current runtime's Python in a separate process, passes the configured
workspace/data roots and disconnects all standard streams. No Tk loop runs in
the MCP server. The receipt means launched, not enrolled; the app owns completion.

## Recording and publication

Selecting a profile/device prepares a continuous mono 24 kHz stream in a worker
thread. Readiness requires PCM, not a WAV header. Recording reuses the process,
includes a 250 ms in-memory pre-roll and caps saved audio at 20 seconds.
Stop flushes the take; Finish/Close stops FFmpeg. Unrecorded audio is discarded.
Blocking device/file work stays off Tk's event thread.

Prepared takes retain stable passage numbers. Raw takes use unique IDs.
Accepted audio/metadata live only in `data\<profile>`; drafts, logs and curation
staging live under the enrollment workspace, which must be outside the data root.
Failed sessions retain scratch state. Successful enrollment removes only the
owning profile's scratch files, not sources or other profiles.

Imports validate the complete merged sample set before copying files; JSON
documents publish atomically. This is not a cross-file transaction or
multi-process profile editor. Reopening retains exact saved transcripts and
does not relabel an imported clip as a different static passage.
Selected checkpoint/speaker/styles survive finishing enrollment.

`Voice` allows extra profile metadata. Defaults are `writing_style="energetic
and informal"`, `frequent_words=[]` and `preferred_greeting=None`.
These fields survive merges but currently do not affect writing or synthesis.
Shared bucket names cannot be profile names; no legacy profile lookup is used.
`YAPIT_DATA_DIR` selects durable data; `YAPIT_WORKSPACE` selects temporary state.


The app manages manifests internally. The lower-level `enroll(store, name,
manifest_path)` function remains for session publication and developer-controlled
imports; it is not the public CLI/MCP contract. Imports merge by normalized filename.

## Example of a generated Enrollment manifest as used by YapIt

```json
{
  "samples": [
    {"audio": "recordings\\take-01.wav", "text": "Replace this with the exact words you spoke in your first recording."},
    {"audio": "recordings\\take-02.wav", "text": "Replace this with the exact words you spoke in your second recording."},
    {"audio": "recordings\\take-03.wav", "text": "Replace this with the exact words you spoke in your third recording."},
    {"audio": "recordings\\take-04.wav", "text": "Replace this with the exact words you spoke in your fourth recording."},
    {"audio": "recordings\\take-05.wav", "text": "Replace this with the exact words you spoke in your fifth recording."}
  ]
}
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest src\enrollment\tests -q
```

Rapid units cover signal gates, name normalization, passage selection and app
launch receipts/configuration with process creation mocked.
Media/profile/recorder/Tk workflows are opt-in under `tests\integration`;
they use synthetic input, not a real microphone. Keep the static JSON passages
packaged. No automated test establishes voice similarity or transcript accuracy.
