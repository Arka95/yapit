# MCP development

Source is `src\mcp`, imported as `yapit.mcp`. See [usage](README.md) and the
[folder-level layout](../dev.readme.md#flat-source-layout).

## Boundaries and entry points

`server.py` owns FastMCP registration, stdio startup, protocol input validation,
path confinement and per-server busy handling. It delegates enrollment to
Enrollment, audio to Yapper, reels to the shared Pipeline and rendering to Editor.
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
The mode launcher is shared application infrastructure, not an MCP-only file.
Its `mcp` branch invokes `python -m yapit.mcp` in the selected runtime rather
than depending on an older environment's generated console wrapper.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest src\mcp\tests -q
```

Tests launch both the module and installed console entry point, initialize a real
stdio client, verify all six tools and the reusable prompt, exercise status and
model-free enrollment, render synthetic audio/timing, and reject confined-path
violations and missing profiles. An isolated subprocess checks lazy package
imports; a resource check verifies the packaged example.

Workspace/data/models are isolated under temporary roots. No microphone,
model download, speech inference, training or GPU workload is run.
Root interface tests now contain only CLI/shared-application coverage.
