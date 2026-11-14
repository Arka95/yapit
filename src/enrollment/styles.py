from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from yapit.common_utils.fileutils import data_filename, save_json
from yapit.common_utils.mediautils import decode_audio, write_wav
from yapit.common_utils.metrics import timed
from yapit.enrollment.profiles import audio_quality, load_voice
from yapit.models import DeliveryStyle, StyleReference, Voice
from yapit.storage import DataStore

__all__ = ["enroll_style", "select_voice", "STYLE_GUIDES"]


STYLE_GUIDES = {
    DeliveryStyle.lively_vlog: {
        "direction": (
            "Sound warmly excited and curious, as if showing a friend something delightful. "
            "Smile naturally, vary emphasis and pitch, and keep the wildlife advice firm but "
            "friendly. No shouting or exaggerated announcer voice."
        ),
        "text": (
            "Look at these little faces! They're curious, fluffy, and full of character. "
            "Let's enjoy the moment, keep a respectful distance, and let them carry on "
            "with their day."
        ),
    },
    DeliveryStyle.relaxed_travel: {
        "direction": (
            "Use a warm, unhurried conversational voice with gentle pitch variation. "
            "Let important words breathe without becoming flat or artificially slow."
        ),
        "text": (
            "Sometimes the best part of a journey is simply slowing down. Take a breath, "
            "look around, and enjoy the small details you might otherwise walk right past."
        ),
    },
    DeliveryStyle.asmr_travel: {
        "direction": (
            "Use a soft, intimate delivery or a comfortable light whisper, with clear words. "
            "Keep the microphone signal usable; avoid clipping, rubbing, or loud breath noise."
        ),
        "text": (
            "Let's stay here for a moment. Listen to the breeze, notice the quiet around us, "
            "and let the rest of the day wait while we take this view in."
        ),
    },
}


@timed
def enroll_style(
    store: DataStore,
    name: str,
    style: DeliveryStyle,
    source: Path,
    text: str,
    consent: bool = True,
) -> StyleReference:
    style = DeliveryStyle(style)
    if not consent:
        raise ValueError("Confirm this is your own voice and the transcript is exact.")
    if style == DeliveryStyle.default:
        raise ValueError("Use enroll-voice for the default voice; choose a named delivery style.")
    try:
        voice = load_voice(store, name)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Profile {name!r} does not exist. "
            "Create it with yapit enroll-app or yapit enroll-voice first."
        ) from exc
    if voice.checkpoint:
        raise ValueError("Reference-driven styles require Base voice cloning, not a checkpoint.")
    if style in voice.styles:
        raise ValueError(
            f"Style {style.value!r} already exists; existing recordings are preserved."
        )
    source = source.resolve()
    if not source.is_file() or source.stat().st_size > 250 * 1024 * 1024:
        raise ValueError("Supply a local audio recording no larger than 250 MB.")
    data = decode_audio(source, max_seconds=20)
    quality = audio_quality(data)
    if quality["reasons"]:
        raise ValueError(" ".join(quality["reasons"]))
    name = store.profile_dir(name).name
    directory = store.voice_audio_dir(name)
    filename = data_filename(style.value, name, ".wav")
    if (directory / filename).exists():
        raise ValueError("The style audio directory exists; refusing to overwrite recordings.")
    words = len(re.findall(r"\b[\w'-]+\b", text))
    reference = StyleReference(
        audio=str(directory / filename),
        text=text,
        words_per_minute=min(210, max(100, words * 60 / quality["seconds"])),
        consent=True,
    )
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="style-", dir=directory.parent) as temporary:
        staged = Path(temporary)
        shutil.copyfile(source, staged / data_filename(style.value, name, f".raw{source.suffix}"))
        write_wav(staged / filename, data)
        shutil.copytree(staged, directory, dirs_exist_ok=True)
    save_json(
        store.profile_dir(name) / data_filename(style.value, name, ".reference.json"),
        {
            **reference.model_dump(mode="json"),
            "style": style.value,
            "direction": STYLE_GUIDES[style]["direction"],
            "quality": quality,
            "method": "human-labeled reference audio; no model training or emotion classification",
        },
    )
    voice.styles[style] = reference
    save_json(store.voice_path(name), voice)
    return reference


def select_voice(
    store: DataStore, name: str, style: DeliveryStyle = DeliveryStyle.default
) -> tuple[Voice, Path]:
    style = DeliveryStyle(style)
    voice = load_voice(store, name)
    if style == DeliveryStyle.default:
        return voice, store.voice_audio_dir(name)
    if voice.checkpoint:
        raise ValueError("Reference-driven styles require Base voice cloning, not a checkpoint.")
    if style not in voice.styles:
        raise ValueError(
            f"No {style.value!r} reference enrolled for {name!r}. Record your own expressive "
            "3-20 second clip and register it with enroll-style first."
        )
    reference = voice.styles[style]
    audio = store.inside(reference.audio)
    if not audio.is_relative_to(store.audio) and not audio.is_relative_to(
        store.voice_audio_dir(name)
    ):
        raise ValueError("Style reference recordings must stay inside the raw audio store.")
    selected = voice.model_copy(
        update={
            "reference_audio": audio.name,
            "reference_text": reference.text,
            "words_per_minute": reference.words_per_minute,
        }
    )
    return selected, audio.parent
