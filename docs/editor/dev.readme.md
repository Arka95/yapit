# Editor development

Source is `src\editor`; public imports are `yapit.editor`. See
[usage and palette](README.md) and the [folder-level project layout](../dev.readme.md#flat-source-layout).

## Responsibilities

`templates.py` selects a background from finite positive cue duration and builds
RGB numbered slates. Green is bright `#00ff00`; darkgreen is `#006400`.
Outlined labels stay legible on light backgrounds.

`render.py` turns a validated continuous Timeline into slates/playlist/SRT, muxes
the existing audio and publishes the encoded video exclusively to its explicit
destination. `__init__.py` exports `render_guide`. Root-source `media.py` is removed;
generic media helpers remain in `common_utils\mediautils.py`.

Root Pipeline and MCP remain application adapters. Audio-derived cue planning
stays in `yapper\timeline.py`; Editor consumes that contract rather than
estimating TTS duration again. The default measured planner still splits longer
beats into short cuts, so not every palette band occurs in every generated reel.

## API and timing

```python
from yapit.editor import render_guide

render_guide(
    timeline,
    audio_path,
    workspace_output,
    video=final_video_path,
    captions=caption_path,
    prefix="arka.trip",
)
```

`Timeline` and paths are supplied by the caller. Application defaults/reservations
remain in `storage.py`; normal workflows use `data\generated_video`.
The renderer's return value is `None`, preserving the existing API.

Colour selection uses the cue's unrounded duration. Encoding uses cumulative
rounded end frames, avoiding independent per-cue rounding drift.
The generated concat playlist specifies `option framerate` for every PNG
(including the terminal repeated image), matching the output fps. This avoids
the image demuxer's default time base shifting clip transitions.
See the official [concat format documentation](https://ffmpeg.org/ffmpeg-formats.html#concat-1)
and [slideshow frame-rate guide](https://trac.ffmpeg.org/wiki/Slideshow).

Per-file options require `-safe 0`; the renderer only creates its own playlist,
restricts prefixes to simple ASCII names and restricts concat input protocols to
local files. It does not accept an external playlist. Shared `run_media` owns
FFmpeg errors/timeouts, and `copy_file_exclusive` prevents overwriting videos.
Publication follows successful encoding; failed encoding publishes nothing.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q src\editor\tests
```

Module tests check every duration boundary, invalid durations, rendered PNG
backgrounds and decoded MP4 RGB frames for all seven bands. The encoded colour
sequence is checked frame-by-frame against cue durations, with only RGB codec
rounding tolerance; synthetic audio duration and publication cleanup are checked
too. Encoding-failure and prefix-injection cases are covered.

Root reel/stdio-MCP tests remain cross-module integration tests. No model
inference, model downloads, cloud requests, microphone sessions or GPU workloads
are needed by Editor's test suite.
