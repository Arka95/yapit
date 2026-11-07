# Editor

Editor renders narration edit guides from a measured `Timeline` and existing
audio. It owns video templates, numbered placeholders, SRT captions and MP4
encoding, not script writing or speech synthesis. See the
[developer guide](dev.readme.md) and [project index](../../README.md).

## Duration-coloured placeholders

Each cue uses a solid background chosen from its actual suggested duration
(`cue.end - cue.start`). Clip number and start/end timing appear in a large,
centered block; the replacement reminder remains smaller near the bottom.
The cue's spoken-script excerpt is smaller, below center, with reserved side
margins and spacing above the footer. Long excerpts wrap to at most four lines;
full text stays available in the SRT captions.

| Cue duration | Base background | Alternate background |
|---|---|---|
| 0 < duration < 1 second | Red | Dark red |
| 1 <= duration < 2 seconds | Cyan | Dark cyan |
| 2 <= duration < 3 seconds | Bright green | Darkgreen |
| 3 <= duration < 4 seconds | Peach | Orange |
| 4 <= duration < 5 seconds | Blue | Dark blue |
| 5 <= duration <= 6 seconds | Yellow | Dark goldenrod |
| duration > 6 seconds | Magenta | Dark magenta |

Only contiguous clips in the same duration band alternate base/darker/base.
Different durations within one band still count as a run. Changing bands resets
the next clip to its base colour; a later return to a band also starts at base.
For example, 2.1s, 2.8s, 2.2s, 3.1s, 3.8s, 2.4s becomes green, darkgreen,
green, peach, orange, green. Exactly 6 seconds belongs to the yellow band.
The colours encode duration, not semantic footage categories.

## Use and outputs

Existing `yapit reel ... --script ...` calls Yapper for narration, then Editor.
The requested video duration is a target with a +/-33% buffer; a 30-second
target permits 20.1-39.9 seconds. The guide follows actual narration timing,
without padding pauses or stretching speech to force an exact length.
After narration, the last visual holds for `--end-buffer` (default 0.75 seconds,
minimum 0.5), rounded up to the next frame. This hold counts in the video
duration range; the original Timeline and captions still describe narration.
MCP `render_timeline` also renders an existing timeline/audio pair confined to
the workspace/data roots. Its existing checks require matching audio duration
and 1080x1920 at 30 fps.

Completed videos go to `data\generated_video\<name>.guide.mp4`, as configured by
`GENERATED_VIDEO_DIRECTORY` in `constants.py`. Captions go to `data\transcripts`.
PNG slates, generated concat playlists and temporary encoding remain under
`workspace\outputs`; the temporary MP4 is removed after successful publication.
Existing videos are not overwritten or automatically migrated.

This is an editing guide, not a finished reel. Replace the coloured placeholders
with your footage. Editor does not generate footage, locate stock clips, choose
the best semantic insertion points or change supplied cue durations.

Rendering uses local Pillow and CPU FFmpeg `libx264`/AAC. It does not load a
speech/script model, call a hosted LLM or configure GPU encoding.

## Optional music and audio controls

**Music is absent by default.** Neither `reel` nor `videogen` searches for or
downloads a track automatically.

```powershell
yapit videogen --timeline workspace\outputs\arka.trip.timeline.json --audio data\generated_audio\arka.trip.mp3
yapit videogen --timeline workspace\outputs\arka.trip.timeline.json --audio data\generated_audio\arka.trip.mp3 --music data\background_music\TRACK.mp3 --audio-volume 100 --music-volume 18 --music-fade-in 0.5 --music-fade-out 1 --end-buffer 0.75
```

`videogen` reuses existing audio and timing; it does not regenerate narration.
`reel` exposes the same mixing controls when synthesizing a new script.
Volumes are linear 0-100 amplitude percentages, not decibels.
**Fade-in/out applies only to music, never narration.** Fades run across the full
video, including its closing hold. Music loops by default; `--no-loop-music`
uses silence after a short track ends.

Speech-driven ducking lowers music during real narration and recovers through
pauses (`--no-duck-music` disables it). Local script/timeline curation supplies
per-beat music levels with 250 ms transitions. `--music-plan FILE` can supply
an edited plan; narration gain remains steady unless its volume is explicitly
changed. A mix limiter bounds summed peaks without automatic gain boosting.
Source narration/music files are not edited in place.

Python callers use `AudioMix` and `generate_video`/`render_guide`; MCP
`generate_reel` and `render_timeline` accept optional `music_path` and an `AudioMix`
`mix` object. Mix settings and file provenance are saved beside job status.
