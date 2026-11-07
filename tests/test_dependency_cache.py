import hashlib
import importlib.util
import io
from pathlib import Path

import pytest


@pytest.fixture
def wheel_cache():
    path = Path(__file__).resolve().parents[1] / "scripts" / "cache-feed-wheel.py"
    spec = importlib.util.spec_from_file_location("cache_feed_wheel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def feed_response(module, monkeypatch, content, downloaded):
    filename = "demo-2.0-py3-none-any.whl"
    checksum = hashlib.sha256(content).hexdigest()
    base = "https://ms-feed-25.pkgs.visualstudio.com/feed/pypi/download/"
    html = (
        f'<a href="{base}demo/1.0/demo-1.0-py3-none-any.whl">older</a>'
        f'<a href="https://files.pythonhosted.org/demo/{filename}#sha256={checksum}">'
        "blocked direct host</a>"
    ).encode()
    calls = []

    def open_url(url, timeout):
        calls.append(url)
        return io.BytesIO(html if url == module.INDEX + "demo/" else downloaded)

    monkeypatch.setattr(module.urllib.request, "urlopen", open_url)
    return filename, base, calls


def test_redirected_wheel_is_verified_and_cached(wheel_cache, monkeypatch, tmp_path):
    content = b"verified artifact"
    filename, base, calls = feed_response(wheel_cache, monkeypatch, content, content)
    path = wheel_cache.cache_wheel(filename, tmp_path)
    assert path.read_bytes() == content
    assert calls == [wheel_cache.INDEX + "demo/", base + "demo/2.0/" + filename]
    assert not list(tmp_path.glob("*.part"))


def test_bad_download_leaves_no_wheel_or_partial(wheel_cache, monkeypatch, tmp_path):
    filename, _, _ = feed_response(wheel_cache, monkeypatch, b"expected", b"modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        wheel_cache.cache_wheel(filename, tmp_path)
    assert not list(tmp_path.iterdir())


def test_modified_existing_wheel_is_not_overwritten(wheel_cache, monkeypatch, tmp_path):
    filename, _, calls = feed_response(wheel_cache, monkeypatch, b"expected", b"expected")
    path = tmp_path / filename
    path.write_bytes(b"local modification")
    with pytest.raises(ValueError, match="Existing cached wheel"):
        wheel_cache.cache_wheel(filename, tmp_path)
    assert path.read_bytes() == b"local modification"
    assert calls == [wheel_cache.INDEX + "demo/"]
