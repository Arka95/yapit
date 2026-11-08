# Editor development

`mixing.py` owns validated `AudioMix`, real-speech sidechain ducking, smooth
per-beat music gain, music-only fades and a minimum 0.5-second closing hold
(default 0.75). `generation.py` publishes a video from existing audio/timing
without speech synthesis or music acquisition. Root CLI/Pipeline/MCP provide
optional locally curated beat levels; Editor itself never searches for music.

`render_guide` returns the frame-rounded video duration after publication.
Its last displayed cue/time and palette include the closing hold; narration
Timeline/SRT end times remain unchanged. Narration is padded only after its end,
not between sentences. Source files are never overwritten. Metadata reports
`seconds` for narration and `video_seconds` for the complete video.

Source is `src\editor`; public imports are `yapit.editor`. See
[usage and palette](README.md) and the [folder-level project layout](../dev.readme.md#flat-source-layout).

## Responsibilities

`templates.py` selects a background from finite positive cue duration and builds
RGB numbered slates. Clip/time labels scale with output width and form a
centered block. Green is bright `#00ff00` with darkgreen `#006400` as its
alternate; the 3-4s band is peach `#ffdab9` with orange `#ffa500` as its alternate.
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
The renderer returns the published video duration as a float.

`RenderSettings` controls codecs, preset, CRF and audio bitrate without changing
the existing defaults. `AudioMix` also exposes ducking parameters, beat-transition
duration and narration/timeline tolerance. `PlaceholderStyle` centralizes layout
defaults; fonts are cached by name/size.

`prepare_audio_inputs` validates files and measures duration; `audio_filters`
builds the graph without file reads and accepts explicit audio input labels.
Callers with a measured narration duration can pass `actual_seconds` to avoid
probing. Otherwise libsndfile reads sample-count metadata, with full decoding
only for unsupported formats. Slate preparation and encoding are separate
stages in `render.py`.

Colour selection uses the cue's unrounded duration. Encoding uses cumulative
rounded end frames, avoiding independent per-cue rounding drift.
The renderer compares base duration-band colours and alternates only within
the current contiguous run. A band change resets alternation, regardless of
the overall clip index or any earlier occurrence of that band.
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

Default tests are unit-only: palette/layout, slate planning, encoding arguments,
timing and graph construction. External processes are mocked. There is no
full-render test.

Real audio-mixing tests live in `src\editor\tests\integration`, are automatically
marked `integration`, and are skipped unless explicitly enabled with
`--run-integration`. Model-dependent tests must also be marked `llm` and
require `--run-llm`. Both flags require `--development-mode=full`; the default
rapid mode skips these checks. These opt-ins apply across the repository.
Unit tests cannot launch subprocesses or network connections.
