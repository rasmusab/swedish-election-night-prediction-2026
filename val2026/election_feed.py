"""Bounded collector for Valmyndigheten's public 2026 Riksdag result feed.

Public specification (accessed 2026-09-05):
https://www.val.se/valresultat-och-statistik/statistik-och-data/teknisk-beskrivning-av-resultatfiler

MD5 detects archive changes; the official certificate and OpenSSL verify every
JSON signature. The certificate is obtained over HTTPS, its fingerprint is
recorded, and it is retained locally. No private media service is used.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile
from zoneinfo import ZoneInfo

BASE_URLS = {
    "rehearsal": "https://resultat.val.se/resultatfiler/genrep2026/",
    "production": "https://resultat.val.se/resultatfiler/val2026/",
}
CERTIFICATE_URL = "https://resultat.val.se/keys/val-sign-crt.pem"
COUNT_NAMES = {"preliminary": "preliminär", "final": "slutlig"}
MAX_ARCHIVE_BYTES = 100_000_000
MAX_JSON_BYTES = 350_000_000


class FeedError(Exception):
    """A feed problem that leaves the last successful snapshot available."""


class FetchError(FeedError):
    def __init__(self, message, *, status=None, retry_after=None, url=None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.url = url


class RetryDeferred(FetchError):
    """A previous response requested a cooldown that still applies this run."""

    def __init__(self, deadline, *, now, status, url=None):
        super().__init__(f"Waiting until {deadline.isoformat()} before contacting the feed again.",
                         status=status, retry_after=(deadline - now).total_seconds(), url=url)
        self.deadline = deadline


class SourceTimestampError(FeedError):
    """A verified archive whose source time cannot safely replace current data."""

    def __init__(self, message, *, kind, candidate, previous):
        super().__init__(message)
        self.kind = kind
        self.candidate = candidate
        self.previous = previous


def utc_now():
    return datetime.now(timezone.utc)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def atomic_write(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def immutable_write(path: Path, content: bytes):
    if path.exists():
        if path.read_bytes() != content:
            raise FeedError(f"Existing immutable snapshot differs: {path}")
    else:
        atomic_write(path, content)


def retry_after_seconds(value, now=None):
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0.0, (date - (now or utc_now())).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def record_retry_deadline(root, error, *, now=None):
    """Share the result host's cooldown across collectors and certificate probes."""
    if error.retry_after is None:
        return None
    now = now or utc_now()
    deadline = (error.deadline if isinstance(error, RetryDeferred)
                else now + timedelta(seconds=error.retry_after))
    path = Path(root) / 'data/collector/retry-not-before.json'
    if path.exists():
        existing = json.loads(path.read_bytes())
        if datetime.fromisoformat(existing['retry_not_before_utc']) >= deadline:
            return existing['retry_not_before_utc']
    value = {'retry_not_before_utc': deadline.isoformat(), 'http_status': error.status,
             'failed_url': error.url}
    atomic_write(path, json_bytes(value))
    return value['retry_not_before_utc']


class HttpClient:
    """Finite retries; a long Retry-After defers instead of retrying too early."""

    def __init__(self, *, opener=urlopen, sleep=time.sleep, max_attempts=3,
                 timeout=45, max_retry_wait=60):
        self.opener = opener
        self.sleep = sleep
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.max_retry_wait = max_retry_wait

    def get(self, url, max_bytes):
        for attempt in range(self.max_attempts):
            try:
                request = Request(url, headers={"User-Agent": "val2026-research-collector/0.1"})
                with self.opener(request, timeout=self.timeout) as response:
                    payload = response.read(max_bytes + 1)
                if len(payload) > max_bytes:
                    raise FeedError(f"HTTP response exceeds {max_bytes:,} bytes: {url}")
                return payload
            except HTTPError as error:
                delay = retry_after_seconds(error.headers.get("Retry-After"))
                if error.code == 404:
                    raise FetchError(
                        "Feed is unavailable (HTTP 404). Production may not yet be published; "
                        "rehearsal files may have been removed. Last good data is retained.",
                        status=404, url=url,
                    ) from error
                retryable = error.code in (429, 500, 502, 503, 504)
                wait = max(delay or 0, 2 ** attempt * 2)
                if not retryable or attempt + 1 == self.max_attempts or wait > self.max_retry_wait:
                    raise FetchError(
                        f"HTTP {error.code} fetching {url}; "
                        f"Retry-After: {delay if delay is not None else 'not supplied'} seconds.",
                        status=error.code, retry_after=delay, url=url,
                    ) from error
                self.sleep(wait)
            except (URLError, TimeoutError, OSError) as error:
                if attempt + 1 == self.max_attempts:
                    raise FetchError(f"Download failed after {self.max_attempts} attempts: {error}", url=url) from error
                self.sleep(min(2 ** attempt * 2, self.max_retry_wait))
        raise FetchError("No download attempts configured")


def parse_index(content):
    entries = {}
    for number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{32}", parts[0]):
            raise FeedError(f"Invalid index entry on line {number}")
        checksum, path = parts
        path = path.removeprefix("*").removeprefix("./")
        parsed = PurePosixPath(path)
        if (not re.fullmatch(r"[A-Za-z0-9_./-]+", path) or parsed.is_absolute()
                or ".." in parsed.parts or path in entries):
            raise FeedError(f"Unsafe or duplicate archive path: {path}")
        entries[path] = checksum.lower()
    if not entries:
        raise FeedError("The result index is empty; no archive is published.")
    return entries


def discover_riksdag(entries, count):
    prefix = "p/rd/" if count == "preliminary" else "s/rd/"
    candidates = [path for path in entries if path.startswith(prefix) and path.endswith("_00_RD.zip")]
    if len(candidates) != 1:
        raise FeedError(f"Expected one {count} national Riksdag archive; found {candidates}")
    return candidates[0]


def read_members(payload):
    members = {}
    with ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise FeedError("Duplicate archive member names")
        for entry in archive.infolist():
            if entry.filename.endswith((".json", "_sign.sha256")):
                limit = MAX_JSON_BYTES if entry.filename.endswith(".json") else 65_536
                if entry.file_size > limit:
                    raise FeedError(f"Oversized archive member: {entry.filename}")
                members[entry.filename] = archive.read(entry)
    json_names = [name for name in members if name.endswith(".json")]
    if not json_names:
        raise FeedError("The archive contains no JSON result files")
    if any(name[:-5] + "_sign.sha256" not in members for name in json_names):
        raise FeedError("A JSON result file is missing its official signature")
    return members


def verify_signatures(members, certificate):
    """Fail closed if OpenSSL, the official certificate, or any signature fails."""
    executable = shutil.which("openssl")
    if not executable:
        raise FeedError("OpenSSL is required to verify the official result signatures")

    def run(*arguments, input=None):
        completed = subprocess.run([executable, *arguments], input=input,
                                   capture_output=True, timeout=45, check=False)
        if completed.returncode:
            raise FeedError("Official signature verification failed: "
                            + completed.stderr.decode(errors="replace").strip()
                            + completed.stdout.decode(errors="replace").strip())
        return completed.stdout

    with tempfile.TemporaryDirectory(prefix="val2026-signature-") as temporary:
        folder = Path(temporary)
        cert_path = folder / "certificate.pem"
        cert_path.write_bytes(certificate)
        run("x509", "-in", str(cert_path), "-noout", "-checkend", "0")
        public_key = run("x509", "-in", str(cert_path), "-pubkey", "-noout")
        key_path = folder / "public-key.pem"
        key_path.write_bytes(public_key)
        verified = []
        for name, content in members.items():
            if not name.endswith(".json"):
                continue
            data_path = folder / "data.json"
            signature_path = folder / "signature.bin"
            data_path.write_bytes(content)
            signature_path.write_bytes(members[name[:-5] + "_sign.sha256"])
            run("dgst", "-sha256", "-verify", str(key_path), "-signature",
                str(signature_path), str(data_path))
            verified.append(name)
        fingerprint = run("x509", "-in", str(cert_path), "-noout", "-fingerprint", "-sha256")
    return {"status": "verified", "json_files_verified": verified,
            "certificate_sha256_fingerprint": fingerprint.decode().strip(),
            "certificate_url": CERTIFICATE_URL}


def validate_result(result, environment, count):
    expected_test = environment == "rehearsal"
    # Official docs say production omits this field. Also accept explicit False,
    # but do not silently interpret null, 0, or strings as a production marker.
    valid_test = (result.get("test") is True if expected_test
                  else "test" not in result or result["test"] is False)
    if not valid_test:
        requirement = "test=True" if expected_test else "an absent test flag or test=False"
        raise FeedError(f"Wrong test flag: {environment} requires {requirement}")
    if result.get("valtyp") != "RD" or result.get("rakningstillfalle") != COUNT_NAMES[count]:
        raise FeedError("Result election or counting stage differs from the requested feed")
    if result.get("valdatum") != "2026-09-13":
        raise FeedError("Unexpected election date in 2026 results")
    districts = result.get("valdistrikt")
    if not isinstance(districts, list):
        raise FeedError("Result contains no district list")
    codes = [district.get("valdistriktskod") for district in districts]
    if any(not isinstance(code, str) or not code for code in codes) or len(codes) != len(set(codes)):
        raise FeedError("Missing or duplicate district codes")
    return districts


def parse_source_timestamp(timestamp):
    """Normalize the feed's offsetless Swedish local timestamps for comparison."""
    if timestamp is None or timestamp == "":
        return None, False
    try:
        source_time = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError) as error:
        raise FeedError(f"Invalid source update timestamp: {timestamp}") from error
    assumed_timezone = source_time.tzinfo is None
    if assumed_timezone:
        source_time = source_time.replace(tzinfo=ZoneInfo("Europe/Stockholm"))
    return source_time, assumed_timezone


def source_freshness(timestamp, now, stale_minutes):
    source_time, assumed_timezone = parse_source_timestamp(timestamp)
    if source_time is None:
        return {"source_age_seconds": None, "source_freshness": "unknown",
                "source_timestamp_timezone_assumption": None,
                "stale_threshold_minutes": stale_minutes}
    age = (now - source_time).total_seconds()
    return {"source_age_seconds": round(age, 1),
            "source_freshness": "future_timestamp" if age < -300 else "stale" if age > stale_minutes * 60 else "recent",
            "source_timestamp_timezone_assumption": "Europe/Stockholm" if assumed_timezone else None,
            "stale_threshold_minutes": stale_minutes}


def source_timestamp_relation(timestamp, previous):
    """Reject time rollback, not numerical count decreases from corrections.

    Equal source timestamps are accepted: different official corrections can
    share the feed's timestamp resolution. Once a dated snapshot is known, an
    absent timestamp cannot be used to bypass the monotonic source-time check.
    """
    candidate_time, _ = parse_source_timestamp(timestamp)
    previous_timestamp = previous.get("source_updated_at")
    previous_time, _ = parse_source_timestamp(previous_timestamp)
    if previous_time is not None:
        if candidate_time is None:
            raise SourceTimestampError(
                "Verified archive has no source update timestamp after a dated snapshot; "
                "the last good snapshot is retained.", kind="source_timestamp_missing",
                candidate=timestamp, previous=previous_timestamp,
            )
        if candidate_time < previous_time:
            raise SourceTimestampError(
                f"Verified archive source timestamp rolled back from {previous_timestamp} "
                f"to {timestamp}; the last good snapshot is retained.",
                kind="source_timestamp_rollback", candidate=timestamp, previous=previous_timestamp,
            )
        return "equal" if candidate_time == previous_time else "newer"
    return "first" if candidate_time is not None else "unknown"


class Collector:
    def __init__(self, root, *, environment="rehearsal", count="preliminary",
                 client=None, verifier=verify_signatures, now=utc_now, stale_minutes=15):
        if environment not in BASE_URLS or count not in COUNT_NAMES:
            raise ValueError("Unknown environment/count")
        self.root = Path(root).resolve()
        self.environment, self.count = environment, count
        self.cache = self.root / "data" / "collector" / environment / count
        self.base_url = BASE_URLS[environment]
        self.client = client or HttpClient()
        self.verifier, self.now = verifier, now
        self.stale_minutes = stale_minutes

    def check(self):
        """Return a receipt; unsuccessful checks never replace latest-success.json.

        ``status`` is HTTP/validation collection success, independent of source
        freshness. Reports should use ``collection_state`` and source timestamps
        too. On errors, ``last_good`` describes retained data with its freshness
        recomputed at this check; it is not a claim that the new fetch succeeded.
        """
        now = self.now()
        checked_at = now.isoformat()
        receipt = {"checked_at_utc": checked_at, "environment": self.environment,
                   "count": self.count, "index_url": urljoin(self.base_url, "index.md5")}
        previous_path = self.cache / "latest-success.json"
        previous = {}
        try:
            if previous_path.exists():
                loaded_previous = json.loads(previous_path.read_bytes())
                if not isinstance(loaded_previous, dict) or loaded_previous.get("status") != "ok":
                    raise FeedError("Stored latest-success receipt is not a valid successful check")
                previous = loaded_previous
            for last_check_path in (self.cache / "latest-check.json",
                                    self.root / 'data/collector/retry-not-before.json'):
                if not last_check_path.exists():
                    continue
                last_check = json.loads(last_check_path.read_bytes())
                if not isinstance(last_check, dict):
                    raise FeedError("Stored latest-check receipt is not a valid JSON object")
                deadline_text = last_check.get("retry_not_before_utc")
                if deadline_text:
                    deadline = datetime.fromisoformat(deadline_text)
                    if deadline.tzinfo is None:
                        raise FeedError("Stored retry deadline must include a timezone")
                    if deadline > now:
                        raise RetryDeferred(deadline, now=now, status=last_check.get("http_status"),
                                            url=last_check.get('failed_url'))
            receipt.update(self._check(now, previous))
        except (FeedError, OSError, ValueError, KeyError, TypeError, BadZipFile,
                subprocess.TimeoutExpired) as error:
            receipt.update({"status": "error", "error": str(error),
                            "error_kind": "validation_error", "collection_state": "error",
                            "last_good_retained": previous_path.exists()})
            if isinstance(error, FetchError):
                state = ("rate_limited" if error.status == 429 else "unavailable")
                receipt.update(http_status=error.status, retry_after_seconds=error.retry_after,
                               failed_url=error.url,
                               collection_state=state,
                               error_kind=f"http_{error.status}" if error.status else "download_error")
                if error.retry_after is not None:
                    receipt["retry_not_before_utc"] = record_retry_deadline(self.root, error, now=self.now())
                if isinstance(error, RetryDeferred):
                    receipt["network_request_skipped"] = True
            elif isinstance(error, SourceTimestampError):
                receipt.update(error_kind=error.kind,
                               collection_state="source_rollback" if error.kind == "source_timestamp_rollback" else "error",
                               candidate_source_updated_at=error.candidate,
                               previous_source_updated_at=error.previous)
            elif isinstance(error, OSError):
                receipt["error_kind"] = "local_io_error"
            if previous:
                retained = {key: previous.get(key) for key in (
                    "checked_at_utc", "source_updated_at", "archive_md5", "archive_sha256",
                    "archive_local", "result_json_local", "test", "districts_counted", "districts_expected",
                )}
                try:
                    retained.update(source_freshness(previous.get("source_updated_at"), self.now(), self.stale_minutes))
                except FeedError:
                    retained.update(source_age_seconds=None, source_freshness="unknown")
                receipt["last_good"] = retained
            else:
                receipt["last_good"] = None
        receipt["check_completed_at_utc"] = self.now().isoformat()
        encoded = json_bytes(receipt)
        name = now.strftime("%Y%m%dT%H%M%S%fZ") + "-" + hashlib.sha256(encoded).hexdigest()[:12]
        immutable_write(self.cache / "checks" / f"{name}.json", encoded)
        atomic_write(self.cache / "latest-check.json", encoded)
        if receipt["status"] == "ok":
            atomic_write(self.cache / "latest-success.json", encoded)
        return receipt

    def _check(self, now, previous):
        index = self.client.get(urljoin(self.base_url, "index.md5"), 2_000_000)
        entries = parse_index(index)
        archive_path = discover_riksdag(entries, self.count)
        checksum = entries[archive_path]
        archive_url = urljoin(self.base_url, archive_path)
        snapshot = self.cache / "snapshots" / checksum
        local_archive = snapshot / PurePosixPath(archive_path).name
        payload = local_archive.read_bytes() if local_archive.exists() else None
        downloaded = payload is None
        if payload is None:
            payload = self.client.get(archive_url, MAX_ARCHIVE_BYTES)
        if hashlib.md5(payload).hexdigest() != checksum:
            raise FeedError("Archive MD5 differs from index (or local cache is corrupt). "
                            "The live index may have changed during download; try again later.")
        members = read_members(payload)
        cert_path = self.cache / "certificate.pem"
        # Refresh the public certificate at least daily; preserve prior snapshots.
        cert_age = now.timestamp() - cert_path.stat().st_mtime if cert_path.exists() else float("inf")
        certificate_cached = cert_age < 86400
        certificate = (cert_path.read_bytes() if certificate_cached
                       else self.client.get(CERTIFICATE_URL, 100_000))
        certificate_refetched_after_failure = False
        try:
            signature = self.verifier(members, certificate)
        except FeedError:
            if not certificate_cached:
                raise
            # A same-day key rotation must not stall collection for 24 hours.
            # Retry exactly once with the official HTTPS certificate; neither
            # a bad archive nor an invalid replacement certificate is accepted.
            certificate = self.client.get(CERTIFICATE_URL, 100_000)
            certificate_cached = False
            certificate_refetched_after_failure = True
            signature = self.verifier(members, certificate)
        vote_names = [name for name in members if "rostfordelning" in name and name.endswith(".json")]
        if len(vote_names) != 1:
            raise FeedError(f"Expected one vote-distribution JSON, found {vote_names}")
        result = json.loads(members[vote_names[0]])
        districts = validate_result(result, self.environment, self.count)
        source_timestamp = result.get("senasteUppdateringstid")
        relation = source_timestamp_relation(source_timestamp, previous)
        changed = previous.get("archive_md5") != checksum
        receipt = {"status": "ok", "collection_state": "updated" if changed else "unchanged",
                   "index_entries": len(entries),
                   "index_sha256": hashlib.sha256(index).hexdigest(),
                   "archive_url": archive_url, "archive_md5": checksum,
                   "archive_sha256": hashlib.sha256(payload).hexdigest(),
                   "archive_downloaded": downloaded,
                   "archive_changed": changed,
                   "archive_bytes": len(payload),
                   "archive_local": str(local_archive.relative_to(self.root)),
                   "result_json_local": str((snapshot / "rostfordelning.json").relative_to(self.root)),
                   "signature": signature, "test": result.get("test", False),
                   "certificate_refetched_after_failure": certificate_refetched_after_failure,
                   "source_test_flag_present": "test" in result,
                   "source_updated_at": source_timestamp,
                   "source_timestamp_relation": relation,
                   "previous_source_updated_at": previous.get("source_updated_at"),
                   "district_records": len(districts),
                   "district_types": dict(Counter(d.get("valdistriktstyp", "unknown") for d in districts)),
                   "districts_reported_by_timestamp": sum(bool(d.get("rapporteringsTid")) for d in districts),
                   "districts_counted": result.get("antalValdistriktRaknade"),
                   "districts_expected": result.get("antalValdistriktSomSkaRaknas"),
                   **source_freshness(source_timestamp, now, self.stale_minutes)}
        # Only commit after checksum, signatures, and schema checks all pass.
        immutable_write(local_archive, payload)
        immutable_write(snapshot / "rostfordelning.json", members[vote_names[0]])
        certificate_sha = hashlib.sha256(certificate).hexdigest()
        immutable_write(self.cache / "certificates" / f"{certificate_sha}.pem", certificate)
        if not certificate_cached:
            atomic_write(cert_path, certificate)
        immutable_write(self.cache / "indexes" / f"{receipt['index_sha256']}.md5", index)
        return receipt
