import pytest

from yapit.common_utils.fileutils import save_json
from yapit.common_utils.mediautils import write_wav
from yapit.enrollment import enroll
from yapit.storage import Workspace
from yapit.tests.helpers import tone


@pytest.fixture
def enrolled(tmp_path):
    workspace = Workspace(tmp_path / "workspace")
    incoming = workspace.data.audio / "incoming"
    incoming.mkdir(parents=True)
    path = incoming / "reference.wav"
    write_wav(path, tone(60))
    manifest = workspace.data.transcripts / "samples.json"
    save_json(manifest, {"samples": [{"audio": str(path), "text": "The reference transcript."}]})
    enroll(workspace.data, "test-voice", manifest, consent=True)
    return workspace
