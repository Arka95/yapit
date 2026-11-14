# Editor development

[Back to YapIt](../../README.md) | [Usage/palette](README.md) |
[Shared test policy](../dev.readme.md#testing-and-diagnostics)

## Ownership and APIs

`templates.py` builds numbered Pillow slates; `mixing.py` owns validated
`AudioMix` and filter planning; `render.py` encodes/publishes; `generation.py`
adapts existing audio/timing into application results. Generic media helpers
stay in `common_utils`; cue planning stays in Yapper.

```python
from yapit.editor import render_guide

video_seconds = render_guide(
    timeline, audio_path, workspace_output,
    video=final_video_path, captions=caption_path, prefix="testuser.trip",
)
```

The caller supplies Timeline/paths/reservations. `RenderSettings` owns codec,
preset/CRF and bitrate; `PlaceholderStyle` owns layout/fonts; `AudioMix` owns
gains, fades, ducking, transition duration and timing tolerance.
`prepare_audio_inputs` validates/measures assets; pure `audio_filters` builds
the graph. Supplied `actual_seconds` avoids another narration probe.

## Timing and publication

Palette selection uses unrounded cue duration; frame planning uses cumulative
rounded end frames to avoid drift. Alternation compares contiguous base bands.
The final visual/palette includes the closing hold, but Timeline/SRT remain
narration-only. Result `seconds` is narration; `video_seconds` includes the hold.

Narration gain stays steady unless explicitly changed. Music-only fades,
speech-sidechain ducking and smooth per-beat levels feed a bounded mix limiter.
The renderer never acquires music or loads a speech model.

Owned concat playlists set each PNG's input framerate, including the repeated
terminal image. Because this needs `-safe 0`, prefixes are simple ASCII names,
playlists are generated internally and allowed protocols are local-only.
Shared `run_media` owns FFmpeg failures/timeouts. Encode before exclusive
publication; failed encoding publishes nothing and source files stay unchanged.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest src\editor\tests -q
```

Rapid units mock external work and cover palette/layout, slate/timing planning,
encoding arguments and filter graphs. Real audio-mixing checks are opt-in
under `tests\integration`. Do not add a full audio-to-MP4 rendering test.
