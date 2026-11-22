from yapit.common_utils.mediautils import write_wav
from yapit.enrollment import EnrollmentSession
from yapit.enrollment.tests.integration.conftest import tone
from yapit.storage import DataStore


def test_real_media_session_resumes_one_paragraph_and_preserves_other_sessions(tmp_path):
    session = EnrollmentSession(DataStore(tmp_path / "data"), "myvoice")
    raw = session.new_take()
    write_wav(raw, tone(60))
    other = EnrollmentSession(session.store, "other")
    other_raw = other.new_take()
    other_raw.write_bytes(b"untouched other-session placeholder")
    session.save_take(raw, True, True)
    resumed = EnrollmentSession(session.store, session.alias)
    voice = resumed.finish()
    assert len(voice.samples) == 1
    assert voice.reference_text == session.draft.readings[0].text
    assert (session.audio_folder / voice.reference_audio).is_file()
    assert other_raw.read_bytes() == b"untouched other-session placeholder"
    assert not raw.exists()
    assert not resumed.path.exists()
    assert EnrollmentSession(session.store, session.alias).draft.completed
