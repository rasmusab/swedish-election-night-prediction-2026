# %%
"""Fetch the public rehearsal index once and cache the preliminary RD archive.

Run: uv run scripts/check-rehearsal-index.py
Run again to exercise the unchanged-file path. No background polling is started.
Source: Valmyndigheten's technical description of 2026 result files:
https://www.val.se/valresultat-och-statistik/statistik-och-data/teknisk-beskrivning-av-resultatfiler

Rehearsal results are TEST DATA. This script never accesses production results.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from zipfile import ZipFile

from val2026 import ROOT
BASE_URL = "https://resultat.val.se/resultatfiler/genrep2026/"
INDEX_URL = urljoin(BASE_URL, "index.md5")
CACHE = ROOT / "data" / "rehearsal-feed"


# %% Helpers: bounded HTTP reads, strict index parsing, and atomic local writes.
def fetch(url, max_bytes):
    request = Request(url, headers={"User-Agent": "val2026-rehearsal-inspection/0.1"})
    with urlopen(request, timeout=45) as response:
        content = response.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise ValueError(f"Response exceeds {max_bytes:,} bytes: {url}")
        return content


def parse_index(content):
    entries = {}
    for number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{32}", parts[0]):
            raise ValueError(f"Invalid index entry on line {number}")
        checksum, raw_path = parts
        path = raw_path.removeprefix("*").removeprefix("./")
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", path):
            raise ValueError(f"Unexpected archive path: {raw_path}")
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise ValueError(f"Unsafe archive path: {raw_path}")
        if path in entries:
            raise ValueError(f"Duplicate archive path: {path}")
        entries[path] = checksum.lower()
    if not entries:
        raise ValueError("The rehearsal index is empty; no archive is available.")
    return entries


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


# %% One index check. Preserve each distinct archive for later replay.
def check_index():
    checked_at = datetime.now(timezone.utc).isoformat()
    content = fetch(INDEX_URL, 2_000_000)
    entries = parse_index(content)
    previous_path = CACHE / "index.md5"
    previous = parse_index(previous_path.read_bytes()) if previous_path.exists() else {}
    added = sorted(entries.keys() - previous.keys())
    removed = sorted(previous.keys() - entries.keys())
    changed = sorted(p for p in entries.keys() & previous.keys() if entries[p] != previous[p])
    print(f"Rehearsal index: {len(entries)} archives")
    print(f"Since previous successful check: {len(added)} added, {len(changed)} changed, {len(removed)} removed")

    # Discover the filename from the index instead of relying on doc examples.
    candidates = [
        p for p in entries
        if p.startswith("p/rd/") and p.endswith("_00_RD.zip")
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one preliminary Riksdag archive; found {candidates}")
    archive_path = candidates[0]
    checksum = entries[archive_path]
    local_archive = CACHE / "snapshots" / checksum / PurePosixPath(archive_path).name
    cached = local_archive.read_bytes() if local_archive.exists() else None
    downloaded = cached is None or hashlib.md5(cached).hexdigest() != checksum
    if downloaded:
        payload = fetch(urljoin(BASE_URL, archive_path), 100_000_000)
    else:
        payload = cached
    if hashlib.md5(payload).hexdigest() != checksum:
        raise ValueError(
            "Archive checksum differs from the index. The feed may have changed "
            "between requests; run again later. The saved index was not updated."
        )

    # Read inside the ZIP without extracting its paths onto disk.
    with ZipFile(io.BytesIO(payload)) as archive:
        matches = [n for n in archive.namelist() if "rostfordelning" in n and n.endswith(".json")]
        if len(matches) != 1:
            raise ValueError(f"Expected one vote-distribution JSON; found {matches}")
        member = matches[0]
        if archive.getinfo(member).file_size > 350_000_000:
            raise ValueError("Vote-distribution JSON exceeds the inspection size limit")
        result = json.loads(archive.read(member))
    if result.get("test") is not True or result.get("valtyp") != "RD":
        raise ValueError("Expected explicitly marked Riksdag rehearsal data")
    if result.get("rakningstillfalle") != "preliminär":
        raise ValueError("Expected preliminary counting data")
    districts = result["valdistrikt"]
    codes = [d["valdistriktskod"] for d in districts]
    if len(codes) != len(set(codes)):
        raise ValueError("Duplicate district codes in rehearsal results")
    district_types = {}
    for district in districts:
        kind = district["valdistriktstyp"]
        district_types[kind] = district_types.get(kind, 0) + 1

    receipt = {
        "checked_at_utc": checked_at,
        "index_url": INDEX_URL,
        "index_entries": len(entries),
        "added": added,
        "changed": changed,
        "removed": removed,
        "archive_url": urljoin(BASE_URL, archive_path),
        "archive_md5": checksum,
        "archive_downloaded": downloaded,
        "archive_bytes": len(payload),
        "archive_local": str(local_archive.relative_to(ROOT)),
        "test": result["test"],
        "source_updated_at": result["senasteUppdateringstid"],
        "district_records": len(districts),
        "district_types": district_types,
        "districts_counted": result["antalValdistriktRaknade"],
        "districts_expected": result["antalValdistriktSomSkaRaknas"],
    }
    # Commit only after all checks succeed. MD5 is consistency checking,
    # not verification of Valmyndigheten's cryptographic signatures.
    if downloaded:
        atomic_write(local_archive, payload)
    atomic_write(CACHE / "index.md5", content)
    receipt_bytes = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    atomic_write(CACHE / "checks" / f"{timestamp}.json", receipt_bytes)
    atomic_write(CACHE / "latest-check.json", receipt_bytes)
    print("Archive downloaded and verified" if downloaded else "Archive unchanged; reused verified local snapshot")
    print("TEST DATA ONLY")
    print(f"Source updated: {receipt['source_updated_at']}")
    print(f"Counted: {receipt['districts_counted']:,} / {receipt['districts_expected']:,}")
    print(f"District types: {district_types}")
    print(f"Receipt: {CACHE / 'latest-check.json'}")
    return receipt


# %% Run once; failures leave the last successful local snapshot intact.
if __name__ == "__main__":
    try:
        receipt = check_index()
    except HTTPError as error:
        if error.code == 429:
            message = f"Rate limited; try later. Retry-After: {error.headers.get('Retry-After', 'not provided')}"
        elif error.code == 404:
            message = "Rehearsal data is unavailable (404); test files may have been removed."
        else:
            message = f"HTTP {error.code}: {error.reason}"
        print(message, file=sys.stderr)
        raise SystemExit(1)
    except (URLError, TimeoutError, ValueError) as error:
        print(f"Check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
