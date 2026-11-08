# Yapper

Yapper owns script planning and audio generation inside YapIt. It accepts supplied
text or structured Script JSON, selects an enrolled voice/style, synthesizes local
narration and exports measured timing. It does not render video or transcribe
source audio. See the [developer guide](dev.readme.md) and [project index](../../README.md).

## Generate narration

For repeated requests, start `yapit voice-worker arka` in your GPU runtime and
leave it running. Normal narration/reel commands with the same data root discover
the authenticated loopback worker and reuse its loaded model. Use
`yapit voice-worker-status` to see readiness, actual device/dtype, model-load
time and last request/audio duration. `yapit voice-worker-stop` releases the
resident model; it is not an OS service.

Unchanged speech is cached privately under `data\.runtime\speech-cache`, bounded
to 250 MiB. Voice/reference or checkpoint file changes invalidate the cache and
reload the single resident model. Set `YAPIT_SPEECH_CACHE=0` to regenerate speech
while retaining the model; set `YAPIT_VOICE_WORKER=0` to explicitly use one-shot
inference. No worker means the existing one-shot behavior. An unavailable
configured worker fails clearly rather than silently loading a second model.
Weights/tokenizers must already be provisioned; worker startup is offline.
Keeping a model loaded avoids startup cost, not the cost of new audio-token
generation. Listening review remains necessary.

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

### Choose a model

`--backend` selects an adapter; `--model` selects the model that adapter loads.
Qwen remains the default, not a requirement:

```powershell
yapit audiogen --text "Leave time to enjoy the view." --voice arka --seconds 5 --backend pocket --model english_2026-10
yapit voice-worker arka --backend pocket --model english_2026-10
```

Pocket TTS requires `.[pocket]` and uses your profile's reference recording on CPU,
not its Qwen checkpoint. `--model` accepts Pocket language/model names or a YAML
config path. An explicit Qwen `--model` overrides the profile's selected checkpoint;
omitting it retains the existing checkpoint/reference behavior.

The worker can switch adapters/models on explicit requests and keeps one loaded
at a time. Without selection flags, requests inherit the worker's startup choice.
`YAPIT_TTS_BACKEND` and `YAPIT_TTS_MODEL` configure defaults. Restart workers started
before this selection API; they reject explicit selectors rather than ignore them.

Other models need their own adapter, registered locally or through an installed
`yapit.tts` plugin. Model weights are not interchangeable between architectures.
See the developer guide for the adapter contract. Optional dependencies and model
assets must be provisioned separately; these commands are not a speed/quality claim.

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
provide delivery, while `pause_ms` specifies inserted silence in addition to
any pauses already present in synthesized speech. Short 5 ms edge ramps soften
waveform joins without trimming words or adding time.

## Script planning

`yapper.script_planner` owns `load_script`, `build_prompt`, `generate_script` and
the packaged `script_prompt.txt`. Existing `yapit prompt` prepares a reusable
brief/schema/word-budget prompt without contacting an LLM.

The separate, explicit `yapit reel ... --auto-script` path can call the optional
cloud script-writing adapter. It requires `.[cloud]`, `YAPIT_ALLOW_CLOUD=true`
and an API key; it sends the brief and optional sampled visual frames, not voice
recordings. Supplied text/JSON narration never invokes that adapter.

## Models, timing and files

The default Qwen adapter uses local Qwen3-TTS Base voice conditioning: one selected
recording and its verified transcript, not automatic weight training. A separately
selected fine-tuned checkpoint uses its named custom speaker instead. CPU
inference is supported; `YAPIT_DEVICE=auto` selects CUDA when available or logs
a CPU fallback once when loading the model. Explicit device choices are preserved.
Model weights may
need downloading if not already available. No general-purpose local script LLM
or production CosyVoice adapter is added by this module.

Target duration is 5-180 seconds. Reels accept natural duration within +/-33%
of the target, without tempo fitting or silence padding; 30 seconds permits
20.1-39.9 seconds. Prioritize useful script content rather than exact length.
Audio-only generation retains the existing +/-5-second tolerance and bounded
0.90-1.10x tempo fitting, including scaling of inserted pauses.
Measured beat timing follows the final encoded narration;
phrase captions/cuts are approximate, not word-level alignment.

Generated WAV/MP3 plus raw/fitted WAV intermediates remain in
`data\generated_audio`; normalized scripts go to `data\transcripts`.
Briefs, provenance, timeline JSON, status and failure logs go to
`workspace\outputs`. Returned JSON contains the canonical file paths.
Existing files are preserved by reserved output names. Video output remains
the separate [Editor](../editor/README.md) responsibility.
