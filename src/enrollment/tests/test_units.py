from types import SimpleNamespace

import numpy as np
import pytest

from yapit.common_utils.fileutils import data_filename
from yapit.common_utils.mediautils import SAMPLE_RATE
from yapit.enrollment import EnrollmentSession, audio_quality
from yapit.models import Sample
from yapit.storage import Workspace


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


@pytest.mark.parametrize("use_configured_workspace", [False, True])
def test_app_launcher_preserves_roots_without_inheriting_protocol_streams(
    tmp_path, monkeypatch, use_configured_workspace
):
    import subprocess
    from unittest.mock import Mock

    from yapit.enrollment import launcher

    configured = tmp_path / "configured-workspace"
    monkeypatch.setenv("YAPIT_WORKSPACE", str(configured))
    root = Workspace(configured if use_configured_workspace else tmp_path / "explicit-workspace")
    launch = Mock(return_value=SimpleNamespace(pid=1234))
    monkeypatch.setattr(launcher.sys, "platform", "win32")
    monkeypatch.setattr(launcher.subprocess, "Popen", launch)
    result = (
        launcher.enroll_voice()
        if use_configured_workspace
        else launcher.enroll_voice(workspace=root)
    )
    command = launch.call_args.args[0]
    settings = launch.call_args.kwargs
    assert command[1:] == ["-m", "yapit.enrollment.desktop"]
    assert settings["env"]["YAPIT_WORKSPACE"] == str(root.root)
    assert settings["env"]["YAPIT_DATA_DIR"] == str(root.data.root)
    assert settings["stdin"] == settings["stdout"] == settings["stderr"] == subprocess.DEVNULL
    assert result["status"] == "launched" and result["pid"] == 1234


def test_app_launcher_rejects_non_windows_without_starting_a_process(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from yapit.enrollment import launcher

    root = Workspace(tmp_path / "workspace")
    launch = Mock()
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    monkeypatch.setattr(launcher.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="interactive Windows desktop"):
        launcher.enroll_voice(workspace=root)
    launch.assert_not_called()
