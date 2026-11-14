# MCP integration

The `yapit.mcp` module exposes YapIt's workflows through Model Context Protocol
(MCP), a tool-call interface used by assistant clients. It runs locally over
stdio, not a public HTTP listener. It is an adapter, not a model provider or
training/generation implementation.

## Install and launch

After [local installation](../../README.md#install-locally-on-windows), from
the repository root:

```powershell
.\yapit.cmd mcp
# Or, from an activated YapIt environment:
python -m yapit.mcp
```

These commands start a server waiting for an MCP client, not an interactive
terminal prompt. `.\yapit.cmd set-mode cpu` or `gpu` sets the runtime once;
no mode argument is needed on MCP calls. Restart an existing client/server to
apply a changed mode. Direct Python invocation uses that interpreter rather than
the saved launcher runtime.

Adapt `src\mcp\examples\mcp.config.json` in your client settings. Replace its
absolute Windows executable/workspace/data paths. The example uses `yapit.cmd mcp`
and the project's saved `YAPIT_MODE`; process environment overrides still win.
GPU dispatch remains strict; use CPU when CUDA is unavailable.

## Tools and execution

| Tool | Purpose |
|---|---|
| `prepare_script_prompt` | Prepare a script-writing brief |
| `generate_script` | Text prompt to Script JSON; `seconds` defaults to 30 |
| `enroll_voice` | Create/update a profile from verified recordings |
| `generate_narration` | Narrate an approved script |
| `generate_reel` | Narration plus vertical edit guide |
| `render_timeline` | Render existing audio and measured timing |
| `get_job_status` | Read a job's saved result/status |

The reusable prompt is `reel_script`. These are MCP tools, not extra CLI commands.

Tool execution has the same model requirements as its underlying workflow:
prompt preparation/enrollment are model-free, narration uses the selected local backend,
and rendering consumes existing audio/timing without loading a speech model.
`generate_script(prompt, seconds=30)` accepts text only, with no image/frame/media
parameters. Cloud writing requires permission and credentials; the full reel
pipeline additionally needs `auto_script=true` to invoke the writer.
Rescript transcription and fine-tuning are not registered MCP tools.

File inputs are confined to configured workspace/data roots. Calls are synchronous;
mutating/generation calls use a per-server busy lock, not a cross-process GPU lock
or durable queue. Review scripts, recordings and generated outputs before use.
Logs must stay on stderr because stdout carries protocol messages.

See [development](dev.readme.md) for implementation/tests and the
[project index](../../README.md) for owning workflow modules.
