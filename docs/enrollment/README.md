# Voice enrollment

[Back to YapIt](../../README.md) | [Developer guide](dev.readme.md)

Create a reusable voice profile from local recordings. Enrollment is model-free:
it records/curates audio but does not transcribe speech or train weights.

## Record your voice

From the clone root, after installation:

```powershell
.\yapit.cmd enroll-voice
```

Create/select a profile, choose a microphone and read the displayed passages.
Replay each take, check **Confirm before saving**, then save. Finish after
**at least five distinct recordings**; more passages can be added later.
Aim for 8-15 seconds in a quiet room with one consistent microphone; takes
are capped at 20 seconds. Exact transcripts matter.

The microphone remains ready between takes; audio is saved only during
**Record**. **Stop** ends the take; **Finish enrollment** or closing the window
releases the microphone. Revisiting a passage lets you replace its prepared take.
Profile names are lowercase/case-insensitive.

## From an agent or Python

```python
from yapit.enrollment import enroll_voice

enroll_voice()
```

MCP `enroll_voice()` opens the same app with no arguments. You do not provide
an MP3, profile name, transcript or manifest; choose the profile and record
inside the window. It launches separately and returns a process ID, not a
completed profile. An interactive Windows desktop is required.

After finishing, check the saved profiles:

```powershell
.\yapit.cmd profiles
```

## Files and limitations

Accepted audio and profile/manifests/quality reports live under
`data\<profile>`, with recordings in `audio`. Raw takes/drafts remain under
`workspace\enrollment\<profile>` until successful enrollment removes them.
Save any wanted raw take before finishing; incomplete sessions remain resumable.

Signal checks reject clipping, excessive silence and unusable recordings;
they do not verify spoken words or speaker identity. The 16 shared passages
are a starter set, not a guarantee of trained voice quality.

Optional profile fields `writing_style`, `frequent_words` and
`preferred_greeting` are preserved metadata, not automatic script/prosody controls.
For expressive audio references, see [Yapper styles](../yapper/README.md#delivery-styles).
Enrollment storage configuration and merge behavior are detailed in the
[developer guide](dev.readme.md).
