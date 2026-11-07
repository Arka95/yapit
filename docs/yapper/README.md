# Yapper

Yapper owns script planning and audio generation inside YapIt. It accepts supplied
text or structured Script JSON, selects an enrolled voice/style, synthesizes local
narration and exports measured timing. It does not render video or transcribe
source audio. See the [developer guide](dev.readme.md) and [project index](../../README.md).

## Generate narration

Run from the repository root after installing the existing `.[voice]` dependencies
and enrolling your own profile:

```powershell
yapit audiogen --text "Take the slower route. Leave time to enjoy the view." --voice arka --seconds 5
yapit audiogen --script data\transcripts\trip.txt --voice arka --seconds 30
yapit audiogen --script data\transcripts\trip.json --voice arka --seconds 30
```

Choose exactly one of `--text` or `--script`. Existing
`yapit narrate --script ...` and MCP `generate_narration` remain supported.
`--style` accepts `default`, `lively-vlog`, `relaxed-travel` or `asmr-travel`;
named styles require a separately enrolled expressive recording.
The old `yapit yapper generate` form remains a hidden compatibility alias;
it is not needed for normal use. Standalone module entry points are documented
in the developer guide.

Register a style once using your own expressive recording and its exact words:

```powershell
yapit enroll-style arka --audio .\excited.wav --text "Look at these little faces! What a lovely surprise." --style lively-vlog
yapit audiogen --text "Today we are exploring the valleys of Ladakh." --voice arka --style lively-vlog --seconds 5
```

The named profile must already exist and validate successfully; create it with
`yapit enroll-app` or `yapit voice-enroll` first. No `--consent` flag is required;
consent defaults to true. Registration saves
the reference under `styles` in `data\arka\arka.profile.json`, keeps prepared/raw
recordings in `data\arka\audio`, and writes
`data\arka\arka.lively-vlog.reference.json`. Future narration with
`--style lively-vlog` loads that saved reference; you do not register it again.
Your default reference remains unchanged, and no model weights are trained.
These paths follow the configured data root and selected profile.

Plain text becomes sentence/paragraph beats with default 180 ms pauses, including
after the final beat. Script JSON permits explicit pauses and visual suggestions:

```json
{
  "title": "Trip",
  "beats": [
    {"text": "Take the slower route.", "visual": "Road footage", "pause_ms": 300},
    {"text": "Leave time to enjoy the view.", "visual": "Landscape", "pause_ms": 0}
  ]
}
```

Words are not rewritten during narration. There is no semantic emphasis
annotator or per-word emphasis field; model prosody and the selected reference
provide delivery, while `pause_ms` specifies inserted silence.

## Script planning

`yapper.script_planner` owns `load_script`, `build_prompt`, `generate_script` and
the packaged `script_prompt.txt`. Existing `yapit prompt` prepares a reusable
brief/schema/word-budget prompt without contacting an LLM.

The separate, explicit `yapit reel ... --auto-script` path can call the optional
cloud script-writing adapter. It requires `.[cloud]`, `YAPIT_ALLOW_CLOUD=true`
and an API key; it sends the brief and optional sampled visual frames, not voice
recordings. Supplied text/JSON narration never invokes that adapter.

## Models, timing and files

Default synthesis uses local Qwen3-TTS Base voice conditioning: one selected
recording and its verified transcript, not automatic weight training. A separately
selected fine-tuned checkpoint uses its named custom speaker instead. CPU
inference is supported; `YAPIT_DEVICE=auto` selects CUDA when available or logs
a CPU fallback once when loading the model. Explicit device choices are preserved.
Model weights may
need downloading if not already available. No general-purpose local script LLM
or production CosyVoice adapter is added by this module.

Target duration is 5-180 seconds with the existing +/-5-second tolerance and
bounded 0.90-1.10x tempo fitting. Fitting affects the whole waveform, including
inserted pauses. Measured beat timing follows the final encoded narration;
phrase captions/cuts are approximate, not word-level alignment.

Generated WAV/MP3 plus raw/fitted WAV intermediates remain in
`data\generated_audio`; normalized scripts go to `data\transcripts`.
Briefs, provenance, timeline JSON, status and failure logs go to
`workspace\outputs`. Returned JSON contains the canonical file paths.
Existing files are preserved by reserved output names. Video output remains
the separate [Editor](../editor/README.md) responsibility.
