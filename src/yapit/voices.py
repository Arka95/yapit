from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from yapit.enrollment.profiles import audio_quality, enroll, load_voice
from yapit.media import decode_audio, write_wav
from yapit.models import DeliveryStyle, StyleReference, Voice
from yapit.storage import Workspace, data_filename, save_json

__all__ = ["audio_quality", "enroll", "enroll_style", "load_voice", "select_voice", "STYLE_GUIDES"]


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


def enroll_style(
    workspace: Workspace,
    name: str,
    style: DeliveryStyle,
    source: Path,
    text: str,
    consent: bool,
) -> StyleReference:
    style = DeliveryStyle(style)
    if not consent:
        raise ValueError("Confirm this is your own voice and the transcript is exact.")
    if style == DeliveryStyle.default:
        raise ValueError("Use voice-enroll for the default voice; choose a named delivery style.")
    voice = load_voice(workspace, name)
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
    name = workspace.profile_dir(name).name
    directory = workspace.profile_dir(name) / "audio"
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
        workspace.profile_dir(name) / data_filename(style.value, name, ".reference.json"),
        {
            **reference.model_dump(mode="json"),
            "style": style.value,
            "direction": STYLE_GUIDES[style]["direction"],
            "quality": quality,
            "method": "human-labeled reference audio; no model training or emotion classification",
        },
    )
    voice.styles[style] = reference
    save_json(workspace.voice_path(name), voice)
    return reference


def select_voice(
    workspace: Workspace, name: str, style: DeliveryStyle = DeliveryStyle.default
) -> tuple[Voice, Path]:
    style = DeliveryStyle(style)
    voice = load_voice(workspace, name)
    if style == DeliveryStyle.default:
        return voice, workspace.voice_audio_dir(name)
    if voice.checkpoint:
        raise ValueError("Reference-driven styles require Base voice cloning, not a checkpoint.")
    if style not in voice.styles:
        raise ValueError(
            f"No {style.value!r} reference enrolled for {name!r}. Record your own expressive "
            "3-20 second clip and register it with voice-style-enroll first."
        )
    reference = voice.styles[style]
    audio = workspace.inside(reference.audio)
    if not audio.is_relative_to(workspace.data.audio) and not audio.is_relative_to(
        workspace.profile_dir(name) / "audio"
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
