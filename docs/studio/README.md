# Desktop studio

[Back to YapIt](../../README.md) | [Developer guide](dev.readme.md)

Open the existing Windows app with `.\yapit.cmd enroll-voice`. This launches the
three-tab studio in your selected runtime; it does not start generation.

| Tab | Workflow |
|---|---|
| Enrollment | Create/open a profile, record the single one-minute paragraph, replay/confirm it, and click **Enroll voice**. |
| Training | Load a profile and collect/reuse its 16 short clips. Choose a Base model and reviewed checkout, then click **Train** in a configured Linux/WSL2 runtime. Select a checkpoint after review. |
| Audio | Select an enrolled profile, TTS backend and a model from **Available models**. Type/paste narration, or enter a prompt and click **Ask AI for text**. Review the text, then click **Generate audio**. |
| Reel | Generated audio and its measured timeline load automatically. Alternatively **Load audio** (up to 180 seconds), then **Analyze speech beats**. Play/pause, seek the waveform or select a narration cut to preview the portrait template. Click **Generate video template**, then **Open generated MP4**. |

AI drafting uses the existing cloud script provider and requires
`YAPIT_ALLOW_CLOUD=true`, credentials and the cloud extra. It never runs
automatically. Audio preserves reviewed words and natural duration. Editing an
AI draft rebuilds its narration beats from the edited text; an unchanged draft
retains its visual suggestions.

The model dropdown discovers local Qwen Base models, compatible trained
checkpoints for the selected voice, and cached clone-capable Pocket variants.
It updates when you change the voice/backend; **Refresh profiles / models**
rescans local files. Configured local models and the selected voice checkpoint
are prioritized. You do not need to enter model IDs or paths. With no compatible
files found, generation is disabled rather than offering an uncached model.
Existing local configuration can also expose a trusted plugin's local model.

The preview shows the audio waveform, narration cuts and synchronized numbered
placeholders, not musical beat detection. Imported-audio analysis runs the
existing offline ASR workflow and requires its assets already installed.
Generated narration reuses its saved timing without transcription. MP4 playback
uses your default Windows video player; it is not embedded in Tk.

Backend discovery does not import speech SDKs. Generation may load/download the
selected model's assets on first use; no model is loaded just by opening tabs.
Finding model weights does not establish that its SDK or every companion asset
is present in the running Python environment.
The app does not install dependencies, change devices, train or merge references.
Output paths are shown in read-only fields; durable audio/video stay in the
existing data directories. Preview WAVs are removed when replaced or on close.
