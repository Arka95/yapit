import json
from pathlib import Path

import numpy as np
import pytest

from yapit.common_utils.mediautils import SAMPLE_RATE, write_wav
from yapit.models import Beat, Script
from yapit.musicgen import download_music, plan_music, search_music
from yapit.musicgen.catalog import CC0, _safe_url
from yapit.musicgen.models import MusicPlan
from yapit.storage import DataStore

ID = "dfa688b2-c121-4f5f-85d8-43e86892d2d9"
SOURCE = "https://freesound.org/people/creator/sounds/651797/"
AUDIO = "https://cdn.freesound.org/previews/651/651797_1391542-hq.mp3"


def record(**updates):
    return {
        "id": ID,
        "title": "Gentle background music loop 100 bpm",
        "creator": "creator",
        "source": "freesound",
        "foreign_landing_url": SOURCE,
        "url": AUDIO,
        "license": "cc0",
        "license_version": "1.0",
        "license_url": CC0,
        "duration": 10000,
        "tags": [{"name": "instrumental"}, {"name": "acoustic"}],
        **updates,
    }


def script():
    return Script(
        title="Trip",
        beats=[
            Beat(text="Walk through the valley at sunset.", visual="Mountain valley"),
            Beat(text="Book early and keep a backup for weather.", visual="Weather reminder"),
        ],
    )


def test_plans_distinguish_scene_energy_and_respect_explicit_mood():
    nature = plan_music(script(), video_type="nature")
    action = plan_music(script(), video_type="adventure")
    assert nature.bpm_max < action.bpm_min
    assert nature.instruments != action.instruments
    assert nature.beats[-1].music_level <= nature.beats[0].music_level
    overridden = plan_music(script(), video_type="my_custom_trip", emotion="energetic")
    assert overridden.video_type == "my_custom_trip"
    assert overridden.emotion == "energetic"
    assert overridden.bpm_min == action.bpm_min


@pytest.mark.parametrize("case", ["reversed_bpm", "duplicate_beat"])
def test_manual_plan_rejects_ambiguous_ranges_and_beat_levels(case):
    values = plan_music(script()).model_dump()
    if case == "reversed_bpm":
        values.update(bpm_min=140, bpm_max=80)
    else:
        values["beats"].append(values["beats"][0].copy())
    with pytest.raises(ValueError):
        MusicPlan.model_validate(values)


def test_search_filters_license_vocals_and_effects_and_deduplicates(monkeypatch):
    calls = []

    def response(url):
        calls.append(url)
        return {
            "results": [
                record(),
                record(license="by"),
                record(license="by-nc"),
                record(title="Singing vocals", tags=[{"name": "music"}, {"name": "vocals"}]),
                record(title="Rain forest field recording", tags=[{"name": "ambient"}]),
            ]
        }

    monkeypatch.setattr("yapit.musicgen.catalog._json", response)
    candidates = search_music(plan_music(script()))
    assert len(candidates) == 1 and candidates[0].id == ID
    assert candidates[0].duration == 10
    assert candidates[0].bpm == 100
    assert len(calls) == 3
    assert all("license=cc0" in url for url in calls)
    assert all("sunset" not in url and "weather" not in url for url in calls)


@pytest.mark.parametrize(
    "url",
    [
        "http://cdn.freesound.org/track.mp3",
        "https://localhost/music.mp3",
        "https://127.0.0.1/music.mp3",
        "https://cdn.freesound.org.evil.example/music.mp3",
        "https://token@cdn.freesound.org/music.mp3",
    ],
)
def test_music_fetches_reject_unapproved_or_credentialed_urls(url):
    with pytest.raises(ValueError, match="HTTPS"):
        _safe_url(url)


def mock_source(monkeypatch, tmp_path, html):
    fixture = tmp_path / "source.wav"
    time = np.arange(10 * SAMPLE_RATE) / SAMPLE_RATE
    write_wav(fixture, (0.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32))
    monkeypatch.setattr("yapit.musicgen.catalog._json", lambda url: record())

    def read(url, *, maximum=0, output=None):
        if url == SOURCE:
            return html.encode()
        assert url == AUDIO
        assert output is not None
        output.write_bytes(fixture.read_bytes())
        return b""

    monkeypatch.setattr("yapit.musicgen.catalog._read", read)
    monkeypatch.setenv("YAPIT_DATA_DIR", str(tmp_path / "data"))
    return DataStore()


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


def test_changed_source_license_cannot_be_overridden_by_related_cc0_link(tmp_path, monkeypatch):
    data = mock_source(
        monkeypatch,
        tmp_path,
        '<a title="Go to the full license text" href="https://creativecommons.org/licenses/by-nc/4.0/">NC</a>'
        '<a title="Go to the full license text" href="https://creativecommons.org/publicdomain/zero/1.0/">CC0</a>',
    )
    with pytest.raises(ValueError, match="does not confirm CC0"):
        download_music(ID, plan_music(script()), data=data)
    assert not list(data.background_music.iterdir())


def test_failed_encoding_does_not_publish_music_or_leave_temporary_files(tmp_path, monkeypatch):
    data = mock_source(
        monkeypatch,
        tmp_path,
        '<a title="Go to the full license text" href="https://creativecommons.org/publicdomain/zero/1.0/">CC0</a>',
    )

    def fail(*args, **kwargs):
        raise RuntimeError("encoding failed")

    monkeypatch.setattr("yapit.musicgen.catalog.run_media", fail)
    with pytest.raises(RuntimeError, match="encoding failed"):
        download_music(ID, plan_music(script()), data=data)
    assert not list(data.background_music.iterdir())
