from __future__ import annotations

import math
import textwrap

from PIL import Image, ImageDraw, ImageFont

from yapit.models import Cue, Timeline


def placeholder_colour(seconds: float, *, alternate: bool = False) -> str:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Placeholder duration must be finite and positive.")
    for limit, colour, darker in (
        (1, "#ff0000", "#8b0000"),
        (2, "#00ffff", "#008b8b"),
        (3, "#00ff00", "#006400"),
        (4, "#ffdab9", "#ffa500"),
        (5, "#0000ff", "#00008b"),
    ):
        if seconds < limit:
            return darker if alternate else colour
    if seconds <= 6:
        return "#b8860b" if alternate else "#ffff00"
    return "#8b008b" if alternate else "#ff00ff"


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def create_placeholder(
    timeline: Timeline, cue: Cue, index: int, *, alternate: bool = False
) -> Image.Image:
    image = Image.new(
        "RGB",
        (timeline.width, timeline.height),
        placeholder_colour(cue.end - cue.start, alternate=alternate),
    )
    draw = ImageDraw.Draw(image)
    stroke = max(1, round(timeline.width / 540))
    labels = (
        (
            f"CLIP {index:02} / {len(timeline.cues):02}",
            _font(max(8, round(timeline.width * 0.095))),
        ),
        (f"{cue.start:.2f}s - {cue.end:.2f}s", _font(max(8, round(timeline.width * 0.070)))),
    )
    tiles = []
    for text, font in labels:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
        tile = Image.new("RGBA", (right - left, bottom - top))
        ImageDraw.Draw(tile).text(
            (-left, -top), text, font=font, fill="white", stroke_width=stroke, stroke_fill="black"
        )
        tiles.append(tile.crop(tile.getbbox()))
    gap = max(8, round(timeline.height * 0.015))
    height = sum(tile.height for tile in tiles) + gap
    y = (timeline.height - height) // 2
    for tile in tiles:
        image.paste(tile, ((timeline.width - tile.width) // 2, y), tile)
        y += tile.height + gap
    script_font = _font(max(8, round(timeline.width * 0.032)))
    line_width = max(4, int(timeline.width * 0.8 / script_font.getlength("W")))
    excerpt = textwrap.fill(
        " ".join(cue.text.split()), width=line_width, max_lines=4, placeholder="..."
    )
    draw.multiline_text(
        (timeline.width / 2, timeline.height * 0.75),
        excerpt,
        anchor="mm",
        align="center",
        spacing=max(4, round(timeline.height * 0.008)),
        font=script_font,
        fill="white",
        stroke_width=stroke,
        stroke_fill="black",
    )
    draw.text(
        (timeline.width / 2, timeline.height * 0.92),
        "Replace with your footage",
        anchor="mm",
        font=_font(max(8, round(timeline.width * 0.026))),
        fill="white",
        stroke_width=stroke,
        stroke_fill="black",
    )
    return image
