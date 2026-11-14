# Voice enrollment

[Back to YapIt](../../README.md) | [Developer guide](dev.readme.md)

Create a reusable voice profile from local recordings. Enrollment is model-free:
it records/curates audio but does not transcribe speech or train weights.

## Record your voice

From the clone root, after installation:

```powershell
.\yapit.cmd enroll-app
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

## Import existing recordings

```powershell
.\yapit.cmd voice-enroll myvoice --manifest .\my-samples.json
.\yapit.cmd profiles
```

A manifest contains verified words for each clip:

```json
{"samples": [{"audio": "take.wav", "text": "The exact words in this recording."}]}
```

Supply at least five distinct usable recordings. Paths may be absolute or
relative to the manifest. Imports merge by normalized filename: new filenames
add clips; matching filenames replace audio/transcript. Sources remain unchanged.

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
