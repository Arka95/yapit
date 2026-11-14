from types import SimpleNamespace

import numpy as np
import pytest

from yapit.common_utils.fileutils import data_filename
from yapit.common_utils.mediautils import SAMPLE_RATE
from yapit.enrollment import EnrollmentSession, audio_quality
from yapit.models import Sample


@pytest.mark.parametrize("data", [np.zeros(4 * SAMPLE_RATE), np.ones(4 * SAMPLE_RATE)])
def test_signal_checks_reject_silence_and_clipping(data):
    assert audio_quality(data)["reasons"]


def test_unified_data_name_handles_case_prefix_and_path_like_input():
    assert data_filename("TESTUSER.My Travel", "TESTUSER", ".WAV") == "testuser.my_travel.wav"
    assert data_filename("..\\TRIP", "TESTUSER", ".SCRIPT.JSON") == "testuser.trip.script.json"
    with pytest.raises(ValueError, match="extension"):
        data_filename("trip", "testuser", ".wav\\escape")


def test_session_records_first_empty_slot_or_explicit_selection():
    session = EnrollmentSession.__new__(EnrollmentSession)
    sample = Sample(audio="saved.wav", text="Saved passage")
    session.draft = SimpleNamespace(readings=["first", "second"], samples=[sample, None])
    session._selected = None
    assert session.index == 1
    assert not session.finished
    session.select_sample(0)
    assert session.index == 0
    with pytest.raises(ValueError, match="Choose a sample"):
        session.select_sample(2)
    session.draft.samples[1] = sample
    assert session.finished
