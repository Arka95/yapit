import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from yapit.common_utils.fileutils import file_sha256
from yapit.storage import DataStore, Workspace


@pytest.mark.parametrize("payload", [b"", b"recording bytes" * 65536], ids=["empty", "multi-chunk"])
def test_recording_archive_hashes_without_loading_entire_file(tmp_path, monkeypatch, payload):
    source = tmp_path / "source.wav"
    source.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    def reject_full_read(path):
        raise AssertionError("Hashing must not load the entire recording with read_bytes.")

    monkeypatch.setattr(Path, "read_bytes", reject_full_read)
    assert file_sha256(source) == expected
    data = DataStore()
    archived = data.archive_recording(source)
    assert archived.parent.name == expected
    assert data.archive_recording(source) == archived
    assert file_sha256(archived) == expected


def test_workspace_and_standalone_modules_share_configured_data_root(tmp_path, monkeypatch):
    configured = tmp_path / "personal-data"
    monkeypatch.setenv("YAPIT_DATA_DIR", str(configured))
    workspace = Workspace(tmp_path / "metadata")
    assert workspace.data.root == DataStore().root == configured
    audio = workspace.data.audio / "source.wav"
    audio.write_bytes(b"fixture")
    assert workspace.inside(audio) == audio
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="inside"):
        workspace.inside(outside)


def test_default_data_root_is_beside_workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("YAPIT_DATA_DIR")
    monkeypatch.setenv("YAPIT_WORKSPACE", str(tmp_path / "workspace"))
    assert Workspace(tmp_path / "workspace").data.root == DataStore().root == tmp_path / "data"


def test_recording_archive_preserves_original_and_reuses_exact_copy(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"original recording bytes")
    data = DataStore()
    archived = data.archive_recording(source)
    assert archived.is_relative_to(data.audio)
    assert archived.read_bytes() == source.read_bytes()
    assert data.archive_recording(source) == archived
    archived.write_bytes(b"changed after import")
    with pytest.raises(ValueError, match="modified"):
        data.archive_recording(source)
    assert source.read_bytes() == b"original recording bytes"


def test_flat_output_reservation_is_safe_and_concurrent(tmp_path):
    workspace = Workspace(tmp_path / "workspace")
    existing = workspace.data.generated_audio / "arka.trip.wav"
    existing.write_bytes(b"existing audio")
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: workspace.output("trip"), range(8)))
    assert {output.name for output in outputs} == {f"arka.trip.{n}" for n in range(1, 9)}
    assert existing.read_bytes() == b"existing audio"
    assert all(output.path("status.json").is_file() for output in outputs)
    unsafe = workspace.output("..\\..\\holiday:take")
    assert unsafe.path("wav", unsafe.audio).parent == workspace.data.generated_audio
    assert "\\" not in unsafe.name and ":" not in unsafe.name
