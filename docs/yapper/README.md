# Yapper

[Back to YapIt](../../README.md) | [Developer guide](dev.readme.md)

Generate local narration and measured timing from supplied text or Script JSON.
Yapper does not render video, transcribe input or rewrite approved words.

## Generate a script

The shared contract is `generate_script(prompt: str, seconds: float = 30) -> Script`.
It accepts text only, not images, frames or media paths.

```powershell
.\yapit.cmd generate-script "A relaxed weekend in Goa" --seconds 30
```

```python
from yapit.yapper import generate_script

script = generate_script("A relaxed weekend in Goa")
```

CLI/MCP return the Script as JSON; Python returns a `Script` object.
This is explicit cloud writing; see [configuration](dev.readme.md#script-planning).

## Generate narration

Enroll a profile first; choose exactly one of `--text` or `--script`:

```powershell
.\yapit.cmd generate-audio --voice myvoice --text "Take the slower route."
.\yapit.cmd generate-audio --voice myvoice --script .\data\transcripts\trip.json --seconds 30
```

Text/TXT input becomes sentence/paragraph beats, each with a default 180 ms
pause, including the last. JSON gives explicit pauses and visuals:

```json
{
  "title": "Trip",
  "beats": [
    {"text": "Take the slower route.", "visual": "Road footage", "pause_ms": 300},
    {"text": "Enjoy the view.", "visual": "Landscape", "pause_ms": 0}
  ]
}
```

Changing text versus JSON is not a continuous-narration quality switch:
each beat is synthesized separately. Omit `--seconds` for natural audio length
with no total duration cap or target validation. An explicit target (5-180
seconds) validates natural speech within +/-33%, without stretching, padding
or truncation. Reels keep their default 30-second target and +/-33% buffer;
the closing hold counts.
Captions are approximate phrase timing, not word alignment.

## Choose a model

`--backend` selects an adapter; `--model` selects that adapter's model.
Qwen is the default when no configured/worker selection overrides it.

| Backend | Requirements and voice source |
|---|---|
| `qwen` | Standard setup; profile reference or selected Qwen checkpoint |
| `pocket` | Optional `.[pocket]` extra/assets/access; CPU reference cloning or exported voice state, not Qwen weights |

```powershell
.\yapit.cmd generate-audio --voice myvoice --text "Enjoy the view." --backend pocket --model english_2026-10
```

`YAPIT_TTS_BACKEND` / `YAPIT_TTS_MODEL` set defaults. An explicit Qwen model
overrides the profile checkpoint; another architecture does not reuse Qwen
weights. Both adapters run in the selected CPU/GPU Python environment; choosing
a model does not create another environment. See
[adapter setup](dev.readme.md#model-onboarding).

## Delivery styles

Named styles (`lively-vlog`, `relaxed-travel`, `asmr-travel`) require a
separately recorded expressive reference on an existing profile:

```powershell
.\yapit.cmd enroll-style myvoice --audio .\excited.wav --text "Look at those little faces!" --style lively-vlog
.\yapit.cmd generate-audio --voice myvoice --text "Look at those little faces!" --style lively-vlog
```

The default reference stays intact; no weights are trained. Qwen's recorded-style
route requires Base cloning, not a selected CustomVoice checkpoint. Put only
spoken words in narration, not delivery directions. References transfer delivery;
they do not guarantee calibrated emotion control.

## Repeated requests

Provision model/tokenizer assets first; worker startup is offline:

```powershell
.\yapit.cmd voice-worker myvoice
.\yapit.cmd voice-worker-status
.\yapit.cmd voice-worker-stop
```

Start the worker in a separate terminal. It keeps one model and a bounded
250 MiB exact-speech cache. `YAPIT_SPEECH_CACHE=0` forces fresh speech without
unloading; `YAPIT_VOICE_WORKER=0` selects one-shot execution. A recorded but
unavailable worker fails rather than silently loading another model.
The worker's device remains fixed until restarted.

One-shot requests load and release their model each time. Model initialization,
weight loading and narration/encoding stages log progress to stderr; completion
logs include elapsed time. Worker reuse avoids repeated model initialization.

`generate-reel --prompt` also uses the text writer before narration and rendering.
Audio/intermediates land in `data\generated_audio`; normalized scripts in
`data\transcripts`; timings/provenance/status in `workspace\outputs`.
