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
    samples = []
    for index in range(5):
        path = incoming / f"take-{index}.wav"
        write_wav(path, tone(4, 180 + index * 30))
        samples.append({"audio": str(path), "text": f"A clear test passage number {index}."})
    manifest = workspace.data.transcripts / "samples.json"
    save_json(manifest, {"samples": samples})
    enroll(workspace.data, "test-voice", manifest, consent=True)
    return workspace
