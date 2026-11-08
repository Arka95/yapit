import json
from pathlib import Path

import pytest

from yapit.musicgen.catalog import download_music
from yapit.musicgen.planner import plan_music
from yapit.musicgen.tests.test_musicgen import ID, SOURCE, mock_source, script


def test_verified_download_normalizes_and_saves_evidence_without_overwrite(tmp_path, monkeypatch):
    data = mock_source(
        monkeypatch,
        tmp_path,
        '<a title="Go to the full license text" href="http://creativecommons.org/publicdomain/zero/1.0/">CC0</a>',
    )
    first = download_music(ID, plan_music(script(), video_type="nature"), data=data)
    second = download_music(ID, plan_music(script(), video_type="nature"), data=data)
    audio = Path(first["audio"])
    assert audio.parent == data.background_music
    assert audio.name.startswith("serene_nature_gentle_background_music")
    assert audio.is_file() and Path(second["audio"]) != audio
    metadata = json.loads(Path(first["metadata"]).read_text())
    assert metadata["candidate"]["license"] == "CC0-1.0"
    assert metadata["license_evidence"]["source_url"] == SOURCE
    assert metadata["sha256"] == first["sha256"]
    assert metadata["seconds"] == pytest.approx(10, abs=0.05)
    assert not list(data.background_music.glob(".music-*"))
