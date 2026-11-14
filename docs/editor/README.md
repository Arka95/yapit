# Editor

[Back to YapIt](../../README.md) | [Developer guide](dev.readme.md)

Render existing narration and a measured Timeline into a numbered edit guide,
MP4 and SRT captions. Editor does not synthesize speech or generate footage.

## Render and mix

```powershell
.\yapit.cmd generate-reel --audio-path data\generated_audio\testuser.trip.mp3
.\yapit.cmd generate-reel --audio-path data\generated_audio\testuser.trip.mp3 --music data\background_music\TRACK.mp3 --music-volume 18
```

Use returned paths for your own job. Audio input reuses the recording and runs
local speech recognition for timing/captions; script/prompt input first generates
narration. Both expose the same mix controls. Music is absent unless selected
explicitly; rendering never searches/downloads tracks. Python
`yapit.editor.generate_video` still accepts an existing Timeline directly.

| Option | Behavior |
|---|---|
| `--audio-volume`, `--music-volume` | Linear 0-100 amplitude, not decibels |
| `--music-fade-in`, `--music-fade-out` | Music-only fades, never narration |
| `--no-loop-music` | Silence after a short track ends |
| `--no-duck-music` | Disable speech-driven music reduction |
| `--end-buffer` | Closing hold: default 0.75 seconds, minimum 0.5 |

Script/prompt reels accept natural duration within +/-33%, including the
frame-rounded closing hold. Audio-input reels preserve the source duration
and add the closing hold. Narration Timeline/SRT remain speech-only.
Videos go to `data\generated_video`, captions to `data\transcripts`, and
temporary slates/playlists to `workspace\outputs`. Sources/existing outputs
are not overwritten. Encoding is CPU Pillow/FFmpeg H.264/AAC.

## Duration-coloured placeholders

Colours describe suggested clip duration, not footage meaning:

| Duration | Base / alternate |
|---|---|
| Under 1s | Red / dark red |
| 1 to under 2s | Cyan / dark cyan |
| 2 to under 3s | Bright green / darkgreen |
| 3 to under 4s | Peach / orange |
| 4 to under 5s | Blue / dark blue |
| 5 to 6s inclusive | Yellow / dark goldenrod |
| Over 6s | Magenta / dark magenta |

Only consecutive cues in one band alternate shades; a band change resets.
Clip/timing labels are centered; excerpts wrap below them, with full text in SRT.
Replace placeholders with your clips; cut suggestions/captions are approximate.
MCP `generate_reel(audio_path=...)` accepts existing audio inside configured
roots and creates a 1080x1920 guide at 30 fps.
