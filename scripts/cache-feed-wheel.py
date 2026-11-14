"""Cache a checksum-verified wheel through this machine's approved package feed."""

import argparse
import hashlib
import re
import urllib.request
import uuid
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

INDEX = "https://packagefeedproxy.microsoft.io/pypi/simple/"
ARTIFACT_HOST = "ms-feed-25.pkgs.visualstudio.com"


class FeedLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.append(dict(attrs).get("href", ""))


def cache_wheel(filename: str, folder: Path) -> Path:
    if Path(filename).name != filename or not filename.endswith(".whl"):
        raise ValueError("Provide a wheel basename, not a path.")
    parts = filename.split("-")
    if len(parts) < 5 or not re.fullmatch(r"[a-zA-Z0-9_.]+", parts[0]):
        raise ValueError("Invalid wheel filename.")
    package = re.sub(r"[-_.]+", "-", parts[0]).lower()
    version = parts[1]
    if not re.fullmatch(r"[a-zA-Z0-9.+_]+", version):
        raise ValueError("Invalid wheel version.")
    parser = FeedLinks()
    with urllib.request.urlopen(INDEX + package + "/", timeout=60) as response:
        parser.feed(response.read().decode("utf-8"))
    matches = [
        urlparse(link)
        for link in parser.links
        if unquote(urlparse(link).path).rsplit("/", 1)[-1] == filename
    ]
    if len(matches) != 1 or not re.fullmatch(r"sha256=[a-fA-F0-9]{64}", matches[0].fragment):
        raise ValueError("The feed must publish exactly one matching wheel with SHA256.")
    expected = matches[0].fragment.removeprefix("sha256=")
    artifact_links = [
        link
        for link in parser.links
        if urlparse(link).hostname == ARTIFACT_HOST and "/pypi/download/" in link
    ]
    if not artifact_links:
        raise ValueError("No approved artifact endpoint was published by the feed.")
    base = artifact_links[0].split("/pypi/download/")[0] + "/pypi/download/"
    url = base + package + "/" + version + "/" + filename
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / filename
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise ValueError("Existing cached wheel does not match the published checksum.")
        return destination
    temporary = destination.with_suffix(f".{uuid.uuid4().hex}.part")
    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=120) as response:
            with temporary.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("Downloaded wheel checksum mismatch.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename")
    parser.add_argument("--folder", type=Path, default=Path(".runtime-wheel-cache"))
    args = parser.parse_args()
    print(cache_wheel(args.filename, args.folder))
