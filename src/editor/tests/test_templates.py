import pytest
from PIL import Image, ImageChops

from yapit.editor.templates import create_placeholder, placeholder_colour
from yapit.models import Cue, Timeline


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.01, "#ff0000"),
        (0.999, "#ff0000"),
        (1, "#00ffff"),
        (1.999, "#00ffff"),
        (2, "#00ff00"),
        (2.999, "#00ff00"),
        (3, "#ffdab9"),
        (3.999, "#ffdab9"),
        (4, "#0000ff"),
        (4.999, "#0000ff"),
        (5, "#ffff00"),
        (6, "#ffff00"),
        (6.001, "#ff00ff"),
    ],
)
def test_duration_bands_have_unambiguous_boundaries(seconds, expected):
    assert placeholder_colour(seconds) == expected


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_duration_is_rejected(seconds):
    with pytest.raises(ValueError, match="finite and positive"):
        placeholder_colour(seconds)


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.5, "#8b0000"),
        (1.5, "#008b8b"),
        (2.5, "#006400"),
        (3.5, "#ffa500"),
        (4.5, "#00008b"),
        (5.5, "#b8860b"),
        (6.5, "#8b008b"),
    ],
)
def test_alternate_shade_stays_in_the_duration_colour_family(seconds, expected):
    assert placeholder_colour(seconds, alternate=True) == expected


def test_clip_number_and_timing_form_a_large_centered_block():
    cue = Cue(start=0, end=2.5, text="Test", visual="Footage", beat=0)
    timeline = Timeline(seconds=2.5, cues=[cue])
    slate = create_placeholder(timeline, cue, 1)
    background = Image.new("RGB", slate.size, placeholder_colour(2.5))
    center = ImageChops.difference(slate, background).crop(
        (0, round(timeline.height * 0.3), timeline.width, round(timeline.height * 0.7))
    )
    box = center.getbbox()
    assert box is not None
    left, top, right, bottom = box
    assert (left + right) / 2 == pytest.approx(timeline.width / 2, abs=2)
    assert (top + bottom) / 2 == pytest.approx(center.height / 2, abs=2)
    assert right - left > timeline.width * 0.4
    assert bottom - top > timeline.height * 0.07


def test_script_excerpt_is_below_center_with_reserved_side_and_footer_margins():
    cue = Cue(
        start=0,
        end=3.5,
        text="Walk through the cave churches, then leave time for the valley at sunset. " * 8,
        visual="Footage",
        beat=0,
    )
    timeline = Timeline(seconds=3.5, cues=[cue])
    slate = create_placeholder(timeline, cue, 1)
    background = Image.new("RGB", slate.size, placeholder_colour(3.5))
    region = ImageChops.difference(slate, background).crop(
        (0, round(timeline.height * 0.6), timeline.width, round(timeline.height * 0.86))
    )
    box = region.getbbox()
    assert box is not None
    left, top, right, bottom = box
    assert left >= timeline.width * 0.08
    assert right <= timeline.width * 0.92
    assert top > 0 and bottom < region.height
    assert bottom - top < timeline.height * 0.12
