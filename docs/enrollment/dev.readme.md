# Enrollment development

[Back to YapIt](../../README.md) | [Usage](README.md) |
[Shared development policy](../dev.readme.md#testing-and-diagnostics)

## Ownership

| Component | Responsibility |
|---|---|
| `profiles.py` | Signal gates, single-reference publication and inventories |
| `styles.py` | Expressive references and voice/style selection |
| `session.py`, `sample_scripts.py`, `reference.json` | Shared recording session, single-paragraph enrollment and exact transcript |
| `artifacts.py` | Profile-owned scratch paths and successful-session cleanup |
| `recorder.py`, `desktop.py` | FFmpeg DirectShow lifecycle, Tk UI and playback |
| `launcher.py` | Independent app launch with configured roots and isolated streams |

Core APIs accept `DataStore`; entry points load `.env`. Shared contracts,
storage, file naming/hashing and media decoding stay in their existing owners.
Enrollment does not import speech SDKs, construct a generation Pipeline or run
ASR/training while recording. The desktop shell also hosts
[Studio tabs](../studio/dev.readme.md), which call generation/recognition only
on explicit user actions. `python -m yapit.enrollment` provides an isolated module CLI.

The public `enroll_voice()` action takes no inputs and opens the existing app.
CLI `enroll-voice` and MCP use this same launcher; `enroll-app` is not an alias.
It starts the current runtime's Python in a separate process, passes the configured
workspace/data roots and disconnects all standard streams. No Tk loop runs in
the MCP server. The receipt means launched, not enrolled; the app owns completion.

## Recording and publication

Selecting a profile/device prepares a continuous mono 24 kHz stream in a worker
thread. Readiness requires PCM, not a WAV header. Recording reuses the process,
includes a 250 ms in-memory pre-roll and uses a workflow-specific limit:
90 seconds for enrollment, 20 seconds for training.
Stop flushes the take; Finish/Close stops FFmpeg. Unrecorded audio is discarded.
Blocking device/file work stays off Tk's event thread.

Enrollment accepts one 45-90 second paragraph. The 16 numbered prompts belong
to `training.sample_scripts`; training collection uses the shared recording
session with independent scratch and publication.

Prepared/raw takes use unique IDs and remain in scratch until publication.
Accepted audio/metadata live only in `data\<profile>`; drafts, logs and curation
staging live under the enrollment workspace, which must be outside the data root.
Failed sessions retain scratch state. Successful enrollment removes only the
owning profile's scratch files, not sources or other profiles.

Imports require exactly one reference and validate it before copying files; JSON
documents publish atomically. This is not a cross-file transaction or
multi-process profile editor. Reopening retains exact saved transcripts and
does not relabel an imported clip as a different static passage.
Published reference filenames include a content digest, so a replacement cannot
overwrite an active recording. Re-enrollment preserves unrelated metadata/styles
and training assets, but clears selected checkpoint/speaker and exported Pocket
state so the new reference is actually used. Checkpoint files remain selectable.

`Voice` allows extra profile metadata. Defaults are `writing_style="energetic
and informal"`, `frequent_words=[]` and `preferred_greeting=None`.
These fields survive re-enrollment but currently do not affect writing or synthesis.
Shared bucket names cannot be profile names; no legacy profile lookup is used.
`YAPIT_DATA_DIR` selects durable data; `YAPIT_WORKSPACE` selects temporary state.


The app manages manifests internally. The lower-level `enroll(store, name,
manifest_path)` function remains for session publication and developer-controlled
imports; it is not the public CLI/MCP contract. Legacy multi-clip samples are
preserved into the separate training store before replacing a profile.

## Example of a generated Enrollment manifest as used by YapIt

```json
{
  "samples": [
    {"audio": "recordings\\reference.wav", "text": "The exact complete paragraph read in this recording."}
  ]
}
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest src\enrollment\tests -q
```

Rapid units mock media and processes, covering single-reference publication,
resuming/replacing a take, legacy-dataset preservation, workflow-specific capture
limits, UI action routing and app launch receipts/configuration.
Media/profile/recorder/Tk workflows are opt-in under `tests\integration`;
they use synthetic input, not a real microphone. Keep the static JSON passages
packaged. No automated test establishes voice similarity or transcript accuracy.
