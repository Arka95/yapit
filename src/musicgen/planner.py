from __future__ import annotations

import re

from yapit.common_utils.metrics import timed
from yapit.models import Script, Timeline
from yapit.musicgen.models import BeatMusic, MusicPlan, slug

# These are editorial starting points, not claims of AI understanding or measured audio features.
PROFILES = {
    "travel": ("uplifting", 90, 115, 0.6, ["acoustic guitar", "light percussion"], "acoustic"),
    "nature": ("serene", 65, 95, 0.3, ["warm pads", "piano", "light percussion"], "ambient"),
    "city_walk": ("upbeat", 95, 120, 0.65, ["bass", "restrained drums", "keys"], "groove"),
    "adventure": ("energetic", 110, 135, 0.85, ["driving drums", "bass", "plucks"], "upbeat"),
    "heritage": ("curious", 75, 105, 0.45, ["acoustic plucks", "airy pads"], "acoustic"),
    "road_trip": ("hopeful", 95, 120, 0.65, ["guitar", "steady drums", "bass"], "indie"),
    "food": ("playful", 90, 115, 0.6, ["plucks", "light percussion", "keys"], "bossa"),
    "luxury": ("elegant", 70, 100, 0.35, ["piano", "soft strings", "mallets"], "piano"),
    "product": ("focused", 90, 115, 0.55, ["clean synth", "restrained drums"], "electronic"),
}
SCENES = {
    "nature": {"sunset", "sunrise", "valley", "mountain", "lake", "forest", "balloon", "balloons"},
    "city_walk": {"street", "streets", "market", "markets", "city", "nightlife"},
    "adventure": {"rafting", "climbing", "surfing", "skiing", "racing", "adventure"},
    "heritage": {"museum", "museums", "churches", "cave", "caves", "underground", "pottery"},
    "road_trip": {"driving", "highway", "roadtrip", "road-trip"},
    "food": {"food", "cooking", "cafe", "restaurant", "tasting"},
    "luxury": {"spa", "resort", "luxury"},
}


@timed
def plan_music(
    source: Script | Timeline, *, video_type: str = "travel", emotion: str | None = None
) -> MusicPlan:
    """Build an editable music brief and per-beat dynamics without network/model calls."""
    grouped: dict[int, list[str]] = {}
    if isinstance(source, Script):
        for index, beat in enumerate(source.beats):
            grouped[index] = [beat.text, beat.visual]
    else:
        for cue in source.cues:
            if cue.beat < 0:
                raise ValueError("Music planning needs non-negative beat identifiers.")
            grouped.setdefault(cue.beat, []).extend([cue.text, cue.visual])
    if not grouped:
        raise ValueError("Supply a script or timeline with beats.")
    requested = slug(video_type, "travel")
    all_words = set(
        re.findall(
            r"[a-z-]+", " ".join(text for texts in grouped.values() for text in texts).lower()
        )
    )
    inferred = max(SCENES, key=lambda kind: len(all_words & SCENES[kind]))
    family = requested
    if requested == "travel" and all_words & SCENES[inferred]:
        family = inferred
    profile = PROFILES.get(family, PROFILES["travel"])
    emotion_family = (
        {
            "serene": "nature",
            "relaxed": "nature",
            "energetic": "adventure",
            "upbeat": "city_walk",
            "hopeful": "road_trip",
            "playful": "food",
            "curious": "heritage",
            "elegant": "luxury",
            "focused": "product",
        }.get(slug(emotion))
        if emotion
        else None
    )
    if emotion_family:
        profile = PROFILES[emotion_family]
    default_emotion, low, high, energy, instruments, genre = profile
    chosen_emotion = slug(emotion) if emotion else default_emotion
    beats = []
    for index, texts in grouped.items():
        words = set(re.findall(r"[a-z-]+", " ".join(texts).lower()))
        scene = max(SCENES, key=lambda kind: len(words & SCENES[kind]))
        matched = bool(words & SCENES[scene])
        scene_profile = PROFILES[scene] if matched else profile
        beat_energy = scene_profile[3] if matched else energy
        if words & {"weather", "backup", "avoid", "warning", "cancelled", "cancels"}:
            beat_energy = min(beat_energy, 0.35)
        beats.append(
            BeatMusic(
                beat=index,
                emotion=chosen_emotion if emotion else scene_profile[0],
                energy=beat_energy,
                music_level=60 + 40 * beat_energy,
                reason=f"{scene if matched else family} scene; leave space for narration",
            )
        )
    return MusicPlan(
        emotion=chosen_emotion,
        video_type=requested,
        bpm_min=low,
        bpm_max=high,
        instruments=instruments,
        queries=[f"{genre} music", "instrumental music", "background music"],
        avoid=["vocals", "speech", "horror", "anthem", "field recording"],
        beats=beats,
    )
