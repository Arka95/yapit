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
| `generate_script` | Text prompt to Script JSON; `seconds` defaults to 30 |
| `generate_audio` | Speak approved Script JSON; defaults to 30 seconds and `testuser` |
| `extract_script` | Local media to reviewable Script JSON and source timings |
| `enroll_voice` | Open the enrollment app; choose a profile and record there |
| `generate_reel` | Exactly one script, existing audio, or text prompt to an edit guide |

Tool execution has the same model requirements as its underlying workflow:
enrollment is model-free, narration uses the selected local backend,
and audio-to-reel uses local speech recognition without generating new voice audio.
`generate_script(prompt, seconds=30)` accepts text only, with no image/frame/media
parameters. Cloud writing requires `YAPIT_ALLOW_CLOUD=true`, `OPENAI_API_KEY`
and the cloud extra.

`enroll_voice()` takes no arguments. It opens the local enrollment window, where
you create/select a profile, choose a microphone, record and review passages, and
finish enrollment. No MP3, transcript or manifest is supplied to the tool.
It returns `status: launched` and a process ID immediately; that is not proof
that enrollment is complete. An interactive Windows desktop is required on the
MCP server machine. The app runs separately without using MCP's input/output streams.

`generate_reel` accepts exactly one of `script`, `audio_path` or `prompt`.
Script input preserves the supplied words; prompt input runs writing, narration
and rendering. Audio input uses offline local ASR to infer timing/captions and
does not require a voice profile, supplied timeline, cloud writer or TTS.
It preserves the source audio without tempo fitting; `seconds` applies only to
script/prompt input. Audio is limited to 180 seconds; the video adds the configured
closing hold (default 0.75 seconds). ASR captions still require review.
Outputs are portrait placeholder edit guides, not finished footage.

`extract_script(media_path, model="small.en", language="en", offline=true)`
returns Script JSON in `scripts[*].content`, alongside each script path and
source timing. Long recordings may yield multiple scripts. No cloud rewriting
is performed; `offline=false` permits model downloads.

The canonical CLI/Python names match the tools:

| CLI | MCP | Python function |
|---|---|---|
| `generate-audio` | `generate_audio` | `yapit.yapper.generate_audio` |
| `generate-script` | `generate_script` | `yapit.yapper.generate_script` |
| `extract-script` | `extract_script` | `yapit.rescript.extract_script` |
| `generate-reel` | `generate_reel` | `yapit.pipeline.generate_reel` |
| `enroll-voice` | `enroll_voice` | `yapit.enrollment.enroll_voice` |

For example:

```powershell
.\yapit.cmd generate-reel --script .\data\transcripts\trip.json --voice myvoice
.\yapit.cmd generate-reel --audio-path .\data\audio\speech.wav
.\yapit.cmd generate-reel --prompt "A relaxed weekend in Goa" --voice myvoice
```

```python
from pathlib import Path
from yapit.pipeline import generate_reel
from yapit.rescript import extract_script

result = extract_script(Path(r"data\audio\speech.wav"))
# Review result["scripts"][0]["content"] against the recording before narration.
reel = generate_reel(audio_path=Path(r"data\audio\speech.wav"))
```

Only these five tools are exposed; there are no helper tools, duplicate aliases
or reusable MCP prompts. Each CLI workflow likewise has one public name.
Fine-tuning and other distinct local utilities are CLI-only.

File inputs are confined to configured workspace/data roots. Calls are synchronous;
mutating/generation calls use a per-server busy lock, not a cross-process GPU lock
or durable queue. Review scripts, recordings and generated outputs before use.
Logs must stay on stderr because stdout carries protocol messages.

See [development](dev.readme.md) for implementation/tests and the
[project index](../../README.md) for owning workflow modules.
