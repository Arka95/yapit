from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from uuid import UUID

from yapit.common_utils.fileutils import copy_file_exclusive, file_sha256, save_json
from yapit.common_utils.mediautils import SAMPLE_RATE, decode_audio, run_media
from yapit.common_utils.metrics import timed
from yapit.musicgen.models import MusicCandidate, MusicPlan, slug
from yapit.storage import DataStore

API = "https://api.openverse.org/v1/audio/"
HOSTS = {
    "api.openverse.org",
    "freesound.org",
    "www.freesound.org",
    "cdn.freesound.org",
    "commons.wikimedia.org",
    "upload.wikimedia.org",
}
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"


def _safe_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in HOSTS
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
    ):
        raise ValueError("Music requests require HTTPS on an approved catalog/source host.")
    return url


class _Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _safe_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _read(url: str, *, maximum: int = 2 * 1024 * 1024, output: Path | None = None) -> bytes:
    request = urllib.request.Request(
        _safe_url(url), headers={"User-Agent": "YapIt/0.1 local music curation"}
    )
    start = time.monotonic()
    payload = bytearray()
    try:
        with urllib.request.build_opener(_Redirects()).open(request, timeout=20) as response:
            _safe_url(response.url)
            while chunk := response.read(64 * 1024):
                payload.extend(chunk)
                if len(payload) > maximum:
                    raise ValueError("Music response exceeds the size limit.")
                if time.monotonic() - start > 90:
                    raise RuntimeError("Music request exceeded its 90-second deadline.")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Music source returned HTTP {exc.code}; retry later.") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("Music source is unavailable; no asset was published.") from exc
    if output:
        output.write_bytes(payload)
    return bytes(payload)


def _json(url: str) -> dict:
    return json.loads(_read(url))


def _cc0_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in {"creativecommons.org", "www.creativecommons.org"}
        and bool(
            re.fullmatch(
                r"/publicdomain/zero/1\.0/?(?:deed\.[a-z-]+/?|legalcode(?:\.[a-z-]+)?/?)?",
                parsed.path,
            )
        )
    )


def _candidate(record: dict, plan: MusicPlan) -> MusicCandidate | None:
    if (
        record.get("license") != "cc0"
        or record.get("license_version") != "1.0"
        or not _cc0_url(record.get("license_url", ""))
        or record.get("source") not in {"freesound", "wikimedia_audio"}
    ):
        return None
    tags = [
        t["name"].lower() for t in record.get("tags", []) if isinstance(t, dict) and "name" in t
    ]
    title = record.get("title", "")
    words = set(re.findall(r"[a-z-]+", f"{title} {' '.join(tags)}".lower()))
    if not words & {"music", "instrumental", "song", "piano", "guitar", "bossa", "acoustic"}:
        return None
    if words & {"vocals", "vocal", "singing", "speech", "rap", "anthem", "horror", "scary"}:
        return None
    duration = record.get("duration")
    duration = float(duration) / 1000 if duration else None
    if duration is not None and not 5 <= duration <= 600:
        return None
    candidate = MusicCandidate(
        id=str(UUID(record["id"])),
        title=title[:500],
        creator=record.get("creator") or "unknown",
        source=record["source"],
        source_url=_safe_url(record["foreign_landing_url"]),
        audio_url=_safe_url(record["url"]),
        license="CC0-1.0",
        license_url=CC0,
        duration=duration,
        tags=tags,
    )
    preferred = set(re.findall(r"[a-z]+", " ".join(plan.instruments + plan.queries).lower()))
    candidate.score = len(words & preferred) * 2
    candidate.reasons = ["CC0 indexed; original source will be rechecked before download"]
    if "instrumental" in words:
        candidate.score += 3
        candidate.reasons.append("instrumental metadata")
    if words & {"dark", "tension", "chase"} and plan.emotion not in {"tense", "energetic"}:
        candidate.score -= 8
        candidate.reasons.append("darker/action sound may conflict with the planned mood")
    match = re.search(
        r"\b(\d{2,3})[- ]?bpm\b|\bbpm[- ]?(\d{2,3})\b", " ".join([title, *tags]), re.I
    )
    if match:
        candidate.bpm = int(match.group(1) or match.group(2))
        if plan.bpm_min <= candidate.bpm <= plan.bpm_max:
            candidate.score += 3
            candidate.reasons.append("declared BPM fits the music brief")
        else:
            candidate.score -= 3
    return candidate


@timed
def search_music(plan: MusicPlan, *, limit: int = 5, page: int = 1) -> list[MusicCandidate]:
    if not 1 <= limit <= 20 or not 1 <= page <= 10:
        raise ValueError("Use limit 1-20 and page 1-10.")
    candidates = {}
    for query in plan.queries[:3]:
        parameters = urllib.parse.urlencode(
            {"q": query[:200], "license": "cc0", "page_size": 20, "page": page, "format": "json"}
        )
        for record in _json(f"{API}?{parameters}").get("results", []):
            try:
                candidate = _candidate(record, plan)
            except (KeyError, ValueError, TypeError):
                continue
            if candidate:
                candidates[candidate.id] = candidate
    return sorted(candidates.values(), key=lambda item: (-item.score, item.id))[:limit]


class _LicenseLink(HTMLParser):
    def __init__(self):
        super().__init__()
        self.href: str | None = None
        self.seen = False

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "a" and values.get("title") == "Go to the full license text" and not self.seen:
            self.seen = True
            if _cc0_url(values.get("href", "")):
                self.href = values["href"]


def _verify_source(candidate: MusicCandidate) -> dict:
    if candidate.source == "freesound":
        parsed = urllib.parse.urlparse(candidate.source_url)
        match = re.fullmatch(r"/people/[^/]+/sounds/(\d+)/?", parsed.path)
        if parsed.hostname not in {"freesound.org", "www.freesound.org"} or not match:
            raise ValueError("Expected a per-sound Freesound source page.")
        sound_id = match.group(1)
        audio = urllib.parse.urlparse(candidate.audio_url)
        if audio.hostname != "cdn.freesound.org" or not re.fullmatch(
            rf"/previews/\d+/{sound_id}_\d+-hq\.mp3", audio.path
        ):
            raise ValueError("Audio URL does not match the licensed Freesound track.")
        payload = _read(candidate.source_url)
        parser = _LicenseLink()
        parser.feed(payload.decode("utf-8"))
        if not parser.href:
            raise ValueError("Original sound page does not confirm CC0; download refused.")
        declaration = parser.href
    else:
        source = urllib.parse.urlparse(candidate.source_url)
        if source.hostname != "commons.wikimedia.org":
            raise ValueError("Expected a Wikimedia Commons source page.")
        params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
        }
        page_id = urllib.parse.parse_qs(source.query).get("curid", [])
        if page_id and page_id[0].isdigit():
            params["pageids"] = page_id[0]
        elif source.path.startswith("/wiki/File:"):
            params["titles"] = urllib.parse.unquote(source.path.removeprefix("/wiki/"))
        else:
            raise ValueError("Expected a per-file Wikimedia source URL.")
        evidence = _json(
            "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
        )
        entries = [
            item
            for page in evidence.get("query", {}).get("pages", {}).values()
            for item in page.get("imageinfo", [])
        ]
        entry = next((item for item in entries if item.get("url") == candidate.audio_url), None)
        declaration = (entry or {}).get("extmetadata", {}).get("LicenseUrl", {}).get("value", "")
        if not _cc0_url(declaration):
            raise ValueError("Original Commons file does not confirm CC0; download refused.")
        payload = json.dumps(evidence, sort_keys=True).encode()
    return {
        "source_url": candidate.source_url,
        "license_url": CC0,
        "source_declaration": declaration,
        "checked_at": datetime.now(UTC).isoformat(),
        "source_response_sha256": hashlib.sha256(payload).hexdigest(),
        "note": "Original source declares CC0; this is evidence, not a non-infringement guarantee.",
    }


@timed
def download_music(candidate_id: str, plan: MusicPlan, *, data: DataStore | None = None) -> dict:
    identifier = str(UUID(candidate_id))
    candidate = _candidate(_json(f"{API}{identifier}/?format=json"), plan)
    if candidate is None:
        raise ValueError("Track lacks acceptable CC0/music metadata; download refused.")
    evidence = _verify_source(candidate)
    root = data or DataStore()
    with tempfile.TemporaryDirectory(prefix=".music-", dir=root.background_music) as temporary:
        raw, prepared = Path(temporary) / "source.audio", Path(temporary) / "prepared.mp3"
        _read(candidate.audio_url, maximum=64 * 1024 * 1024, output=raw)
        run_media(
            [
                "-y",
                "-protocol_whitelist",
                "file,pipe",
                "-i",
                str(raw),
                "-vn",
                "-t",
                "601",
                "-af",
                "loudnorm=I=-23:TP=-2:LRA=7",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(prepared),
            ],
            timeout=180,
        )
        duration = len(decode_audio(prepared, max_seconds=600)) / SAMPLE_RATE
        base = (
            f"{slug(plan.emotion)}_{slug(plan.video_type)}_{slug(candidate.title)}_{identifier[:8]}"
        )
        number = 0
        while True:
            name = f"{base}_{number}" if number else base
            output = root.background_music / f"{name}.mp3"
            metadata = output.with_suffix(".json")
            if metadata.exists():
                number += 1
                continue
            try:
                copy_file_exclusive(prepared, output)
                break
            except FileExistsError:
                number += 1
        result = {
            "status": "downloaded_needs_listening_review",
            "audio": str(output),
            "metadata": str(metadata),
            "seconds": duration,
            "sha256": file_sha256(output),
            "source_audio_sha256": file_sha256(raw),
            "candidate": candidate.model_dump(mode="json"),
            "license_evidence": evidence,
            "music_plan": plan.model_dump(mode="json"),
            "source_format": "Freesound HQ preview"
            if candidate.source == "freesound"
            else "source file",
        }
        try:
            save_json(metadata, result)
        except Exception:
            output.unlink(missing_ok=True)
            raise
        return result
