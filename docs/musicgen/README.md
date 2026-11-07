# Musicgen

Musicgen prepares script-aware music briefs, searches for CC0 music, rechecks the
original source and downloads a selected track. **Music is optional and off by
default** in video generation. AI music generation is deferred by explicit choice;
this module is not Meta's MusicGen model.
Search is powered by Openverse; YapIt is not endorsed or certified by Openverse.

See [development](dev.readme.md), [design](design.md) and the
[project index](../../README.md).

## Find music for a script

```powershell
yapit music-plan --script data\transcripts\trip.json --video-type nature --output data\transcripts\trip.music-plan.json
yapit music-search --script data\transcripts\trip.json --video-type nature --limit 5
yapit music-download --id CANDIDATE_UUID --script data\transcripts\trip.json --video-type nature
```

Text and Script JSON are accepted. `--emotion` overrides the overall mood.
Nine editorial profiles cover general travel, nature, city walks, adventure,
heritage, road trips, food, luxury and product videos. Other video-type names are
accepted with general-travel defaults. The plan suggests BPM, instruments,
generic search terms and restrained per-beat music levels.

Ranking uses source metadata, declared BPM and instrumental/mood cues; it is
not an audio-understanding model or a guarantee of artistic fit. Preview
candidate tracks before using them. Instrumental metadata is preferred and
obvious vocals, speech, horror/anthem tracks and non-musical recordings are
filtered. Full scripts, private profiles and narration are not sent to search
providers; only the plan's generic music queries leave the machine.
Inspect search terms before submitting a manually customized plan.

## Licensing and files

Only **CC0 1.0** is currently accepted. General royalty-free, attribution-required,
non-commercial and unknown licenses are rejected. Discovery uses Openverse,
then the original Freesound per-track page or Wikimedia per-file metadata is
checked before any audio download. Missing, changed or ambiguous source evidence
fails closed. Unsupported providers are not silently trusted.

CC0 is a public-domain dedication, not a universal legal warranty. Source
declarations cannot prove that an uploader owns every underlying right or
prevent erroneous platform claims. See the official
[CC0 terms](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en) and
[Openverse licensing warning](https://docs.openverse.org/terms_of_service.html).
Public-domain-marked assets are not yet supported because their recording and
composition rights need separate verification.

Audio lands in `data\background_music`, using
`{emotion}_{videotype}_{songname_or_id}.mp3`. A short source ID is included in
the song component; collisions add a numeric suffix instead of overwriting.
Matching JSON records creator/source links, CC0 evidence and its retrieval time,
source/prepared SHA-256 hashes, actual duration and the music plan.

Freesound downloads use its publicly available HQ MP3 preview, not the
authentication-protected original master; this limitation is recorded.
Tracks are prepared as stereo 48 kHz/192 kbps MP3 with a -23 LUFS normalization
target. They are not silently shortened, regenerated or registered as a voice.

## Use music explicitly

```powershell
yapit reel arka --script data\transcripts\trip.json --music data\background_music\TRACK.mp3 --music-volume 18
yapit videogen --timeline workspace\outputs\arka.trip.timeline.json --audio data\generated_audio\arka.trip.mp3 --music data\background_music\TRACK.mp3 --music-plan data\transcripts\trip.music-plan.json
```

Omit `--music` for narration-only video audio. Search/download are separate,
explicit operations; rendering never triggers them. Mixing, looping, music-only
fades and speech ducking belong to [Editor](../editor/README.md).
