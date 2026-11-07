import pytest

from yapit.editor.templates import placeholder_colour


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.01, "#ff0000"),
        (0.999, "#ff0000"),
        (1, "#00ffff"),
        (1.999, "#00ffff"),
        (2, "#00ff00"),
        (2.999, "#00ff00"),
        (3, "#006400"),
        (3.999, "#006400"),
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
