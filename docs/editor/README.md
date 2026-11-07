# Editor

Editor renders narration edit guides from a measured `Timeline` and existing
audio. It owns video templates, numbered placeholders, SRT captions and MP4
encoding, not script writing or speech synthesis. See the
[developer guide](dev.readme.md) and [project index](../../README.md).

## Duration-coloured placeholders

Each cue uses a solid background chosen from its actual suggested duration
(`cue.end - cue.start`), with readable clip/time/replacement labels:

| Cue duration | Background |
|---|---|
| 0 < duration < 1 second | Red |
| 1 <= duration < 2 seconds | Cyan |
| 2 <= duration < 3 seconds | Bright green |
| 3 <= duration < 4 seconds | Darkgreen |
| 4 <= duration < 5 seconds | Blue |
| 5 <= duration <= 6 seconds | Yellow |
| duration > 6 seconds | Magenta |

Darkgreen fills the previously unspecified 3-4-second band. Exact boundary
behaviour is defined above; exactly 6 seconds is yellow.
The colours encode duration, not semantic footage categories.

## Use and outputs

Existing `yapit reel ... --script ...` calls Yapper for narration, then Editor.
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
