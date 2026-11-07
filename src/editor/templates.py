from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

from yapit.models import Cue, Timeline


def placeholder_colour(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Placeholder duration must be finite and positive.")
    for limit, colour in (
        (1, "#ff0000"),
        (2, "#00ffff"),
        (3, "#00ff00"),
        (4, "#006400"),
        (5, "#0000ff"),
    ):
        if seconds < limit:
            return colour
    return "#ffff00" if seconds <= 6 else "#ff00ff"


def create_placeholder(timeline: Timeline, cue: Cue, index: int) -> Image.Image:
    image = Image.new(
        "RGB", (timeline.width, timeline.height), placeholder_colour(cue.end - cue.start)
    )
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default(size=28)
    labels = (
        ((64, 160), f"CLIP {index:02} / {len(timeline.cues):02}"),
        ((64, 220), f"{cue.start:.2f}s - {cue.end:.2f}s"),
        ((64, timeline.height - 230), "Replace with your footage"),
    )
    for position, text in labels:
        draw.text(position, text, font=font, fill="white", stroke_width=1, stroke_fill="black")
    return image
