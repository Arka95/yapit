from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from yapit.models import Cue, Timeline

DURATION_BANDS = (
    (1, "#ff0000", "#8b0000"),
    (2, "#00ffff", "#008b8b"),
    (3, "#00ff00", "#006400"),
    (4, "#ffdab9", "#ffa500"),
    (5, "#0000ff", "#00008b"),
)


@dataclass(frozen=True)
class PlaceholderStyle:
    font: str = "DejaVuSans.ttf"
    clip_size: float = 0.095
    time_size: float = 0.070
    script_size: float = 0.032
    footer_size: float = 0.026
    label_gap: float = 0.015
    script_width: float = 0.8
    script_lines: int = 4
    script_y: float = 0.75
    footer_y: float = 0.92
    line_spacing: float = 0.008
    footer: str = "Replace with your footage"


DEFAULT_STYLE = PlaceholderStyle()


def placeholder_colour(seconds: float, *, alternate: bool = False) -> str:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Placeholder duration must be finite and positive.")
    for limit, colour, darker in DURATION_BANDS:
        if seconds < limit:
            return darker if alternate else colour
    if seconds <= 6:
        return "#b8860b" if alternate else "#ffff00"
    return "#8b008b" if alternate else "#ff00ff"


@lru_cache(maxsize=32)
def _font(size: int, name: str = DEFAULT_STYLE.font) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _draw_labels(
    image: Image.Image, timeline: Timeline, cue: Cue, index: int, style: PlaceholderStyle
) -> None:
    draw = ImageDraw.Draw(image)
    stroke = max(1, round(timeline.width / 540))
    labels = (
        (
            f"CLIP {index:02} / {len(timeline.cues):02}",
            _font(max(8, round(timeline.width * style.clip_size)), style.font),
        ),
        (
            f"{cue.start:.2f}s - {cue.end:.2f}s",
            _font(max(8, round(timeline.width * style.time_size)), style.font),
        ),
    )
    tiles = []
    for text, font in labels:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
        tile = Image.new("RGBA", (right - left, bottom - top))
        ImageDraw.Draw(tile).text(
            (-left, -top), text, font=font, fill="white", stroke_width=stroke, stroke_fill="black"
        )
        tiles.append(tile.crop(tile.getbbox()))
    gap = max(8, round(timeline.height * style.label_gap))
    height = sum(tile.height for tile in tiles) + gap
    y = (timeline.height - height) // 2
    for tile in tiles:
        image.paste(tile, ((timeline.width - tile.width) // 2, y), tile)
        y += tile.height + gap


def create_placeholder(
    timeline: Timeline,
    cue: Cue,
    index: int,
    *,
    alternate: bool = False,
    style: PlaceholderStyle = DEFAULT_STYLE,
) -> Image.Image:
    image = Image.new(
        "RGB",
        (timeline.width, timeline.height),
        placeholder_colour(cue.end - cue.start, alternate=alternate),
    )
    _draw_labels(image, timeline, cue, index, style)
    draw = ImageDraw.Draw(image)
    stroke = max(1, round(timeline.width / 540))
    script_font = _font(max(8, round(timeline.width * style.script_size)), style.font)
    line_width = max(4, int(timeline.width * style.script_width / script_font.getlength("W")))
    excerpt = textwrap.fill(
        " ".join(cue.text.split()),
        width=line_width,
        max_lines=style.script_lines,
        placeholder="...",
    )
    draw.multiline_text(
        (timeline.width / 2, timeline.height * style.script_y),
        excerpt,
        anchor="mm",
        align="center",
        spacing=max(4, round(timeline.height * style.line_spacing)),
        font=script_font,
        fill="white",
        stroke_width=stroke,
        stroke_fill="black",
    )
    draw.text(
        (timeline.width / 2, timeline.height * style.footer_y),
        style.footer,
        anchor="mm",
        font=_font(max(8, round(timeline.width * style.footer_size)), style.font),
        fill="white",
        stroke_width=stroke,
        stroke_fill="black",
    )
    return image
