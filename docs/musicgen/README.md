# Musicgen

[Back to YapIt](../../README.md) | [Developer guide](dev.readme.md)

Prepare music briefs, discover CC0 tracks and download a selected source-verified
track. Music remains optional/off by default. This is not Meta's MusicGen
model; no AI music generation runtime is installed.

## Find and select music

```powershell
.\yapit.cmd music-plan --script data\transcripts\trip.json --video-type nature --output data\transcripts\trip.music-plan.json
.\yapit.cmd music-search --script data\transcripts\trip.json --video-type nature --limit 5
.\yapit.cmd music-download --id CANDIDATE_UUID --script data\transcripts\trip.json --video-type nature
```

Text and Script JSON are accepted. `--emotion` overrides mood. Local editorial
rules suggest tempo/instruments and beat levels; search ranking uses metadata,
not audio understanding. Preview candidates for artistic fit.
Only generic music queries leave the machine, not private narration/profiles.

## Licensing and files

Only **CC0 1.0** is accepted. Openverse discovers candidates; the original
Freesound/Wikimedia declaration must be verified before downloading.
Missing/ambiguous/changed evidence fails closed. Source declarations are not
a universal rights warranty; see [CC0](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en)
and [Openverse's warning](https://docs.openverse.org/terms_of_service.html).

Audio and source/license/hash/plan JSON land in `data\background_music`, named
`{emotion}_{videotype}_{songname_or_id}.mp3`; collisions gain a suffix.
Freesound uses its public HQ preview, not the protected original master.
Prepared MP3 is stereo 48 kHz/192 kbps at a -23 LUFS target.

```powershell
.\yapit.cmd reel myvoice --script data\transcripts\trip.json --music data\background_music\TRACK.mp3 --music-volume 18
```

Rendering never triggers acquisition. Looping, fades, beat modulation and
speech ducking belong to [Editor](../editor/README.md).
