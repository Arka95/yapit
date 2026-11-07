import pytest

from yapit.common_utils.mediautils import SAMPLE_RATE
from yapit.models import Beat, Script
from yapit.yapper.timeline import build_timeline

from .conftest import tone


def test_long_beat_prefers_quiet_window_and_keeps_cues_contiguous():
    audio = tone(8)
    audio[round(3.75 * SAMPLE_RATE) : round(3.85 * SAMPLE_RATE)] = 0
    one = Script(title="Long beat", beats=[Beat(text="One two three four five six seven eight.")])
    timeline = build_timeline(one, [8], 8, audio)
    assert 3.75 <= timeline.cues[0].end <= 3.85
    assert timeline.cues[0].end == timeline.cues[1].start
    assert timeline.cues[-1].end == 8
    with pytest.raises(ValueError, match="positive"):
        build_timeline(one, [0], 8)
