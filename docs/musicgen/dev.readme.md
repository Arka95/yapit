# Musicgen development

Source is `src\musicgen`, imported as `yapit.musicgen`. This is a CPU/network
module with no additional dependencies, speech/model initialization or AI runtime.
See [usage](README.md), [project](../../README.md) and
[source ownership](../dev.readme.md#flat-source-layout).

| File | Responsibility |
|---|---|
| `models.py` | Music briefs, beat dynamics, candidate metadata and filename-safe tags |
| `planner.py` | Local, editable scene/mood/tempo/instrument curation from Script or Timeline |
| `catalog.py` | Bounded HTTPS discovery, original-source CC0 checking and media publication |
| `cli.py` | Flat music-plan/search/download operations |
| `tests` | License/source rejection, selection/privacy, normalization and publication failures |

`plan_music(source, video_type=..., emotion=...)` takes Script or measured
Timeline and returns `MusicPlan`. `search_music(plan, limit=5, page=1)` sends
at most three generic Openverse queries and ranks deduplicated CC0 candidates.
`download_music(candidate_id, plan, data=...)` refetches the UUID detail,
verifies the original source, downloads and normalizes before exclusive publication.
No render path calls these network functions.

Openverse durations are milliseconds; candidate durations are seconds.
Category metadata is often missing, so search does not rely solely on
`category=music`; non-musical results are checked locally instead.
Declared BPM is metadata, not a measured estimate. Editorial rules deliberately
avoid inferring instruments from a destination's nationality.

Only approved HTTPS hosts are fetched, including redirects. URLs cannot carry
credentials. Metadata/source responses are limited to 2 MiB; audio to 64 MiB,
with 20-second socket timeouts and a 90-second elapsed deadline. FFmpeg uses
local file/pipe protocols when processing untrusted downloads. Prepared tracks
over 600 seconds are refused. No HTTP/API credentials or HTML login tokens
are retained in sidecars.

Freesound requires the first per-track license link to declare CC0 and the
preview URL's sound ID to match the original source page. Wikimedia requires
the original per-file `imageinfo` URL and `LicenseUrl` to match. Index metadata
alone is not accepted as download evidence.

Prepared audio is published with exclusive creation. Sidecars use shared atomic
JSON writes; handled sidecar failures remove only the newly owned output.
A process crash between audio and sidecar publication can leave an orphan;
there is no cross-file transaction. Existing audio and metadata are never reused
or overwritten automatically.

```powershell
.\.venv\Scripts\python.exe -m pytest -q src\musicgen\tests src\editor\tests src\tests\test_storage.py
.\.venv\Scripts\python.exe -m ruff check src\musicgen src\editor
```

No training test, longest-clip preflight, speech generation or paid cloud call
is part of this module's validation.
