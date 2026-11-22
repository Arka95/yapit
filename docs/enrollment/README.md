# Voice enrollment

[Back to YapIt](../../README.md) | [Developer guide](dev.readme.md)

Create a reusable voice profile from one continuous recording of a fixed,
phonetically varied English paragraph. Reading time is approximately one minute.

## Record your voice

From the clone root, after installation:

```powershell
.\yapit.cmd enroll-voice
```

This opens the desktop studio. Create/select a profile, choose a microphone and
read the whole paragraph naturally in a quiet room. Aim for about 60 seconds;
accepted recordings are 45-90 seconds. Replay the take, check **Confirm before
saving**, then click **Enroll voice**. Its exact paragraph and audio become the
profile's single reference.

The passage is `sampleScripts\sample_1.json`; the saved recording is
`data\<profile>\audio\<profile>.audio_sample_1.wav`. Re-recording replaces this
same passage slot after confirmation.

**Record** begins capture; **Stop** ends the take. Enrollment completion or
leaving the tab releases the microphone. **Record replacement** repeats the
same paragraph and preserves the published reference until enrollment succeeds.
Profile names are lowercase/case-insensitive. A saved, unfinished take resumes
when the profile is reopened.

The [Training tab](../training/README.md) owns the 16 short fine-tuning passages.
The [Audio and Reel tabs](../studio/README.md) provide narration, waveform/cut
preview and video-template generation.

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
`data\<profile>`, with reference recordings in `audio`. Raw takes/drafts remain under
`workspace\enrollment\<profile>\reference` until successful enrollment removes them.
Save any wanted raw take before finishing; incomplete sessions remain resumable.

Signal checks reject clipping, excessive silence and unusable recordings;
they do not verify spoken words or speaker identity. Enrollment performs no model
loading, transcription or training. Phonetic variety and duration alone cannot
guarantee speaker likeness.

Existing multi-clip profiles remain readable. Their recordings can be separated
into `data\<profile>\training` while retaining the current reference and
checkpoints. Record the new paragraph to complete the new enrollment workflow.
Re-enrollment selects the new reference: an old checkpoint/Pocket voice-state
selection is cleared, while checkpoint files and training recordings remain.

Optional profile fields `writing_style`, `frequent_words` and
`preferred_greeting` are preserved metadata, not automatic script/prosody controls.
For expressive audio references, see [Yapper styles](../yapper/README.md#delivery-styles).
Enrollment storage configuration and publication behavior are detailed in the
[developer guide](dev.readme.md).
