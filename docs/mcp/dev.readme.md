# MCP development

Source is `src\mcp`, imported as `yapit.mcp`. See [usage](README.md) and the
[folder-level layout](../dev.readme.md#flat-source-layout).

## Boundaries and entry points

`server.py` owns FastMCP registration, stdio startup, protocol input validation,
path confinement and per-server busy handling. It delegates enrollment to
Enrollment, audio to Yapper, extraction to Rescript, reels to the shared Pipeline
and rendering to Editor. It calls those owning modules directly, without an API
facade. CLI names use hyphens where Python/MCP use underscores.
Shared contracts/configuration/storage remain outside MCP; do not duplicate them.

`python -m yapit.mcp` and `yapit-mcp` start the same server. The console entry point
targets `yapit.mcp.server:main`. `src\server.py` is removed; callers of the old
`python -m yapit.server` must update. Importing `yapit.mcp` alone initializes neither
the MCP SDK/server nor speech models.

The external SDK is top-level `mcp`; this project's package is `yapit.mcp`.
The names are distinct. Ruff explicitly treats `mcp` as a third-party dependency.
The `mcp` extra stays optional for ordinary CLI/audio users.

`examples\mcp.config.json` belongs to this module and is included in the wheel.
It is also accessible through `importlib.resources.files("yapit.mcp")`.
The root `yapit.cmd mcp` launcher uses the saved `YAPIT_MODE`. Its implementation
is shared application infrastructure, not an MCP-only file.
Its `mcp` branch invokes `python -m yapit.mcp` in the selected runtime rather
than depending on an older environment's generated console wrapper.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest src\mcp\tests -q
```

Rapid units mock generation, ASR and rendering, and verify exactly the five
requested workflows, source dispatch, defaults and path confinement.
Shared Pipeline tests verify audio input uses source timestamps without voice
loading or resynthesis. Extraction includes actual Script JSON in its response.

Opt-in external tests in `tests\integration\test_mcp.py` launch both entry points,
initialize a stdio client, and reject confined-path violations and missing
profiles. They use the protocol, not application imports, and never open the GUI.
Tool inventory/schema maintenance checks are local-only under ignored `agenttests`.

Workspace/data/models are isolated under temporary roots. No microphone,
model download, speech inference, training or GPU workload is run.
Shared CLI units live in `src\tests`; root `tests` is exclusively black-box integration.

`generate_reel` rejects zero/multiple sources. Script/prompt input reuses the
existing narrated pipeline. Audio input calls offline Rescript with a 180-second
limit before recognition, builds contiguous cues from ASR segment starts through
the full decoded duration, and delegates to Editor with the original input path.
No TTS, voice selection, text rewriting or tempo fitting runs on this branch.
Leading/trailing silence remains; the closing hold is a video-only addition.

`enroll_voice()` delegates to Enrollment's launcher and returns immediately.
The child uses the same interpreter/runtime and configured workspace/data roots,
with detached lifecycle and no inherited protocol streams. The existing app
owns profile selection, microphone use, review and completion; a launch receipt
does not claim a profile was created. Unit tests mock process/GUI effects.
