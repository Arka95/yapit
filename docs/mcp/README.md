# MCP integration

The `yapit.mcp` module exposes YapIt's workflows through Model Context Protocol
(MCP), a tool-call interface used by assistant clients. It runs locally over
stdio, not a public HTTP listener. It is an adapter, not a model provider or
training/generation implementation.

## Install and launch

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[mcp]"
.\.venv\Scripts\yapit-mcp
# Equivalent module entry point:
.\.venv\Scripts\python.exe -m yapit.mcp
```

These commands start a server waiting for an MCP client, not an interactive
terminal prompt. After updating an editable checkout, reinstall into each runtime
to refresh the `yapit-mcp` console entry point. The old `python -m yapit.server`
path has been replaced.

Adapt `src\mcp\examples\mcp.config.json` in your client settings. Replace its
absolute Windows executable/workspace/data paths and choose `cpu` or `gpu`
explicitly. The example uses the shared runtime launcher in `scripts`.
GPU dispatch remains strict; use CPU when CUDA is unavailable.

## Tools and execution

The six tools are `prepare_script_prompt`, `enroll_voice`, `generate_narration`,
`generate_reel`, `render_timeline` and `get_job_status`. The reusable prompt is
`reel_script`. See the [main inventory](../../README.md#mcp-and-helper-commands)
for their purposes. These names are MCP tools, not additional CLI commands.

Tool execution has the same model requirements as its underlying workflow:
prompt preparation/enrollment are model-free, narration uses local Qwen,
and rendering consumes existing audio/timing without loading a speech model.
Cloud script writing requires explicit `auto_script`, permission and credentials.
Rescript transcription and fine-tuning are not registered MCP tools.

File inputs are confined to configured workspace/data roots. Calls are synchronous;
mutating/generation calls use a per-server busy lock, not a cross-process GPU lock
or durable queue. Review scripts, recordings and generated outputs before use.
Logs must stay on stderr because stdout carries protocol messages.

See [development](dev.readme.md) for implementation/tests and the
[project index](../../README.md) for owning workflow modules.
