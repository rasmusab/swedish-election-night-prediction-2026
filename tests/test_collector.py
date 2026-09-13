"""Collector failure/recovery and cryptographic integrity checks; no network."""
from datetime import datetime, timedelta, timezone
from email.message import Message
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError
from zipfile import ZIP_DEFLATED, ZipFile

from val2026.election_feed import (
    BASE_URLS, CERTIFICATE_URL, Collector, FeedError, FetchError, HttpClient,
    parse_index, retry_after_seconds, source_freshness, verify_signatures,
)


ARCHIVE_PATH = "p/rd/Genrep_2026_preliminar_00_RD.zip"
MEMBER = "Genrep_2026_preliminar_rostfordelning_00_RD.json"


def result(test=True, vote=42, **changes):
    value = {
        "test": test, "valtyp": "RD", "rakningstillfalle": "preliminär",
        "valdatum": "2026-09-13", "senasteUppdateringstid": "2026-09-01T12:00:00",
        "antalValdistriktRaknade": 1, "antalValdistriktSomSkaRaknas": 2,
        "valdistrikt": [{"valdistriktskod": "01010001", "valdistriktstyp": "valdistrikt",
                        "rapporteringsTid": "2026-09-01T12:00:00", "totaltAntalRoster": vote,
                        "extraMetadataForFutureModel": {"preserve": True}}],
    }
    value.update(changes)
    return value


def archive(data, signature=True):
    stream = io.BytesIO()
    with ZipFile(stream, "w", ZIP_DEFLATED) as zipped:
        zipped.writestr(MEMBER, json.dumps(data).encode())
        if signature:
            zipped.writestr(MEMBER[:-5] + "_sign.sha256", b"mock-signature")
    return stream.getvalue()


class FakeClient:
    def __init__(self, payload, environment="rehearsal", path=ARCHIVE_PATH):
        self.base = BASE_URLS[environment]
        self.path = path
        self.calls = []
        self.set_payload(payload)

    def set_payload(self, payload):
        self.payload = payload
        self.index = f"{hashlib.md5(payload).hexdigest()} ./{self.path}\n".encode()

    def get(self, url, max_bytes):
        self.calls.append(url)
        if url == self.base + "index.md5":
            return self.index
        if url == self.base + self.path:
            return self.payload
        if url == CERTIFICATE_URL:
            return b"mock official certificate"
        raise AssertionError(f"Unexpected URL {url}")


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.client = FakeClient(archive(result()))
        self.verifier = Mock(return_value={"status": "verified", "json_files_verified": [MEMBER]})
        self.collector = Collector(self.root, client=self.client, verifier=self.verifier)

    def test_unchanged_reuses_archive_and_preserves_all_district_metadata(self):
        first = self.collector.check()
        self.assertEqual(first["status"], "ok")
        self.assertTrue(first["archive_downloaded"])
        self.client.calls.clear()
        second = self.collector.check()
        self.assertEqual(second["status"], "ok")
        self.assertFalse(second["archive_downloaded"])
        self.assertFalse(second["archive_changed"])
        self.assertEqual(self.client.calls, [self.client.base + "index.md5"])
        self.assertEqual(self.verifier.call_count, 2)
        raw = json.loads((self.root / second["result_json_local"]).read_bytes())
        self.assertTrue(raw["valdistrikt"][0]["extraMetadataForFutureModel"]["preserve"])
        self.assertEqual(len(list((self.collector.cache / "checks").glob("*.json"))), 2)

    def test_changed_archive_retains_first_snapshot_and_advances_pointer(self):
        first = self.collector.check()
        old_archive = self.root / first["archive_local"]
        old_bytes = old_archive.read_bytes()
        self.client.set_payload(archive(result(vote=123)))
        second = self.collector.check()
        self.assertEqual(second["status"], "ok")
        self.assertTrue(second["archive_changed"])
        self.assertTrue(second["archive_downloaded"])
        self.assertEqual(old_archive.read_bytes(), old_bytes)
        self.assertNotEqual(first["archive_local"], second["archive_local"])
        latest = json.loads((self.collector.cache / "latest-success.json").read_bytes())
        self.assertEqual(latest["archive_md5"], second["archive_md5"])

    def test_checksum_race_preserves_last_success(self):
        first = self.collector.check()
        good_pointer = (self.collector.cache / "latest-success.json").read_bytes()
        self.client.set_payload(archive(result(vote=123)))
        self.client.index = b"00000000000000000000000000000000 ./" + ARCHIVE_PATH.encode() + b"\n"
        failed = self.collector.check()
        self.assertEqual(failed["status"], "error")
        self.assertIn("MD5 differs", failed["error"])
        self.assertTrue(failed["last_good_retained"])
        self.assertEqual((self.collector.cache / "latest-success.json").read_bytes(), good_pointer)
        self.assertTrue((self.root / first["archive_local"]).exists())
        self.assertEqual(len(list((self.collector.cache / "snapshots").iterdir())), 1)

    def test_bad_signature_preserves_last_success(self):
        self.collector.check()
        good_pointer = (self.collector.cache / "latest-success.json").read_bytes()
        self.client.set_payload(archive(result(vote=123)))
        self.verifier.side_effect = FeedError("Invalid official signature")
        failed = self.collector.check()
        self.assertEqual(failed["status"], "error")
        self.assertEqual((self.collector.cache / "latest-success.json").read_bytes(), good_pointer)

    def test_missing_signature_fails_before_verification(self):
        self.client.set_payload(archive(result(), signature=False))
        failed = self.collector.check()
        self.assertIn("missing its official signature", failed["error"])
        self.verifier.assert_not_called()

    def test_cached_key_rotation_refetches_once_and_saves_verified_replacement(self):
        self.collector.check()
        cert_path = self.collector.cache / "certificate.pem"
        cert_path.write_bytes(b"old key")
        self.client.calls.clear()
        self.verifier.side_effect = [FeedError("Signature mismatch with old key"),
                                     {"status": "verified", "json_files_verified": [MEMBER]}]
        receipt = self.collector.check()
        self.assertEqual(receipt["status"], "ok")
        self.assertTrue(receipt["certificate_refetched_after_failure"])
        self.assertEqual(self.client.calls.count(CERTIFICATE_URL), 1)
        self.assertEqual(cert_path.read_bytes(), b"mock official certificate")

    def test_invalid_replacement_key_does_not_replace_certificate_or_last_success(self):
        self.collector.check()
        cert_path = self.collector.cache / "certificate.pem"
        cert_path.write_bytes(b"previous verified key")
        pointer = (self.collector.cache / "latest-success.json").read_bytes()
        self.client.calls.clear()
        self.verifier.side_effect = FeedError("Invalid signature with both keys")
        failed = self.collector.check()
        self.assertEqual(failed["status"], "error")
        self.assertEqual(self.client.calls.count(CERTIFICATE_URL), 1)
        self.assertEqual(cert_path.read_bytes(), b"previous verified key")
        self.assertEqual((self.collector.cache / "latest-success.json").read_bytes(), pointer)

    def test_fresh_key_verification_failure_is_not_refetched(self):
        self.verifier.side_effect = FeedError("Invalid official signature")
        failed = self.collector.check()
        self.assertEqual(failed["status"], "error")
        self.assertEqual(self.client.calls.count(CERTIFICATE_URL), 1)
        self.assertFalse((self.collector.cache / "certificate.pem").exists())

    def test_production_rejects_rehearsal_and_malformed_test_flag(self):
        for test_value in (True, None, 0):
            with self.subTest(test=test_value):
                client = FakeClient(archive(result(test=test_value)), environment="production")
                collector = Collector(self.root, environment="production", client=client, verifier=self.verifier)
                failed = collector.check()
                self.assertEqual(failed["status"], "error")
                self.assertIn("Wrong test flag", failed["error"])
                self.assertFalse((collector.cache / "latest-success.json").exists())

    def test_production_accepts_officially_documented_omitted_test_flag(self):
        data = result()
        del data["test"]
        client = FakeClient(archive(data), environment="production")
        collector = Collector(self.root, environment="production", client=client, verifier=self.verifier)
        receipt = collector.check()
        self.assertEqual(receipt["status"], "ok")
        self.assertFalse(receipt["test"])
        self.assertFalse(receipt["source_test_flag_present"])

    def test_explicit_production_and_final_have_separate_storage(self):
        self.collector.check()
        client = FakeClient(archive(result(test=False)), environment="production")
        production = Collector(self.root, environment="production", client=client, verifier=self.verifier)
        receipt = production.check()
        self.assertEqual(receipt["status"], "ok")
        self.assertFalse(receipt["test"])
        final_path = "s/rd/Val_2026_slutlig_00_RD.zip"
        final_client = FakeClient(archive(result(test=False, rakningstillfalle="slutlig")),
                                 environment="production", path=final_path)
        final = Collector(self.root, environment="production", count="final", client=final_client,
                          verifier=self.verifier)
        self.assertEqual(final.check()["status"], "ok")
        self.assertNotEqual(final.cache, production.cache)
        self.assertNotEqual(production.cache, self.collector.cache)

    def test_wrong_count_or_duplicate_districts_is_rejected(self):
        bad_count = result(rakningstillfalle="slutlig")
        duplicates = result()
        duplicates["valdistrikt"] *= 2
        for value in (bad_count, duplicates):
            self.client.set_payload(archive(value))
            self.assertEqual(self.collector.check()["status"], "error")

    def test_certificate_age_is_not_reset_by_cache_hits(self):
        self.collector.check()
        cert_path = self.collector.cache / "certificate.pem"
        initial_time = cert_path.stat().st_mtime
        self.collector.check()
        self.assertEqual(cert_path.stat().st_mtime, initial_time)
        os.utime(cert_path, (initial_time - 100_000, initial_time - 100_000))
        self.client.calls.clear()
        self.collector.check()
        self.assertIn(CERTIFICATE_URL, self.client.calls)

    def test_signed_timestamp_rollback_preserves_last_good_snapshot_and_receipt(self):
        self.collector.now = Mock(return_value=datetime(2026, 9, 1, 10, 1, tzinfo=timezone.utc))
        good = self.collector.check()
        pointer = (self.collector.cache / "latest-success.json").read_bytes()
        self.client.set_payload(archive(result(vote=123, senasteUppdateringstid="2026-09-01T11:59:59")))
        self.collector.now.return_value += timedelta(minutes=30)
        failed = self.collector.check()
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["collection_state"], "source_rollback")
        self.assertEqual(failed["error_kind"], "source_timestamp_rollback")
        self.assertEqual(failed["candidate_source_updated_at"], "2026-09-01T11:59:59")
        self.assertEqual(failed["previous_source_updated_at"], good["source_updated_at"])
        self.assertEqual(failed["last_good"]["archive_md5"], good["archive_md5"])
        self.assertEqual(failed["last_good"]["source_freshness"], "stale")
        self.assertEqual(failed["last_good"]["source_age_seconds"], 31 * 60)
        self.assertEqual((self.collector.cache / "latest-success.json").read_bytes(), pointer)
        self.assertEqual(len(list((self.collector.cache / "snapshots").iterdir())), 1)
        self.assertEqual(self.verifier.call_count, 2)  # Signature success cannot bypass rollback policy.

    def test_vote_and_count_decreases_are_allowed_with_equal_or_newer_timestamp(self):
        first = self.collector.check()
        self.assertEqual(first["source_timestamp_relation"], "first")
        for timestamp, relation, vote in (("2026-09-01T12:00:00", "equal", 30),
                                           ("2026-09-01T12:01:00", "newer", 20)):
            with self.subTest(relation=relation):
                self.client.set_payload(archive(result(vote=vote, senasteUppdateringstid=timestamp,
                                                       antalValdistriktRaknade=0)))
                corrected = self.collector.check()
                self.assertEqual(corrected["status"], "ok")
                self.assertEqual(corrected["collection_state"], "updated")
                self.assertEqual(corrected["source_timestamp_relation"], relation)
                raw = json.loads((self.root / corrected["result_json_local"]).read_bytes())
                self.assertEqual(raw["valdistrikt"][0]["totaltAntalRoster"], vote)
                self.assertEqual(corrected["districts_counted"], 0)

    def test_equivalent_timezone_spelling_is_not_rollback(self):
        self.collector.check()  # 12:00 Swedish local = 10:00 UTC in September.
        self.client.set_payload(archive(result(vote=20, senasteUppdateringstid="2026-09-01T10:00:00+00:00")))
        corrected = self.collector.check()
        self.assertEqual(corrected["status"], "ok")
        self.assertEqual(corrected["source_timestamp_relation"], "equal")

    def test_missing_source_timestamp_cannot_replace_dated_snapshot(self):
        self.collector.check()
        pointer = (self.collector.cache / "latest-success.json").read_bytes()
        self.client.set_payload(archive(result(vote=123, senasteUppdateringstid=None)))
        failed = self.collector.check()
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["error_kind"], "source_timestamp_missing")
        self.assertEqual((self.collector.cache / "latest-success.json").read_bytes(), pointer)

    def test_initial_undated_feed_has_unknown_freshness_until_timestamp_exists(self):
        self.client.set_payload(archive(result(senasteUppdateringstid=None)))
        initial = self.collector.check()
        self.assertEqual(initial["status"], "ok")
        self.assertEqual(initial["source_timestamp_relation"], "unknown")
        self.assertEqual(initial["source_freshness"], "unknown")
        self.client.set_payload(archive(result()))
        dated = self.collector.check()
        self.assertEqual(dated["status"], "ok")
        self.assertEqual(dated["source_timestamp_relation"], "first")

    def test_network_failure_reports_retained_data_and_preserves_success_pointer(self):
        self.collector.now = Mock(return_value=datetime(2026, 9, 1, 10, 1, tzinfo=timezone.utc))
        good = self.collector.check()
        pointer = (self.collector.cache / "latest-success.json").read_bytes()
        self.collector.now.return_value += timedelta(minutes=30)
        self.client.get = Mock(side_effect=FetchError("No network connection"))
        failed = self.collector.check()
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["collection_state"], "unavailable")
        self.assertEqual(failed["error_kind"], "download_error")
        self.assertEqual(failed["last_good"]["checked_at_utc"], good["checked_at_utc"])
        self.assertEqual(failed["last_good"]["source_freshness"], "stale")
        self.assertEqual((self.collector.cache / "latest-success.json").read_bytes(), pointer)
        saved_error = json.loads((self.collector.cache / "latest-check.json").read_bytes())
        self.assertEqual(saved_error["collection_state"], "unavailable")

    def test_rate_limited_before_first_snapshot_has_explicit_retry_and_no_last_good(self):
        now = datetime(2026, 9, 1, 10, 1, tzinfo=timezone.utc)
        self.collector.now = Mock(return_value=now)
        self.client.get = Mock(side_effect=FetchError("Rate limited", status=429, retry_after=900))
        failed = self.collector.check()
        self.assertEqual(failed["collection_state"], "rate_limited")
        self.assertEqual(failed["error_kind"], "http_429")
        self.assertIsNone(failed["last_good"])
        self.assertFalse(failed["last_good_retained"])
        self.assertEqual(failed["retry_not_before_utc"], (now + timedelta(seconds=900)).isoformat())

    def test_unchanged_check_does_not_make_source_fresh(self):
        self.collector.now = Mock(return_value=datetime(2026, 9, 1, 10, 1, tzinfo=timezone.utc))
        first = self.collector.check()
        self.assertEqual(first["source_freshness"], "recent")
        self.collector.now.return_value += timedelta(hours=1)
        unchanged = self.collector.check()
        self.assertEqual(unchanged["status"], "ok")
        self.assertEqual(unchanged["collection_state"], "unchanged")
        self.assertEqual(unchanged["source_timestamp_relation"], "equal")
        self.assertEqual(unchanged["source_freshness"], "stale")
        self.assertEqual(unchanged["source_updated_at"], first["source_updated_at"])

    def test_cooldown_survives_new_collectors_without_moving_the_deadline(self):
        now = datetime(2026, 9, 1, 10, 1, tzinfo=timezone.utc)
        self.collector.now = lambda: now
        self.collector.check()
        pointer = (self.collector.cache / "latest-success.json").read_bytes()
        get = self.client.get
        self.client.get = Mock(side_effect=FetchError("Rate limited", status=429, retry_after=900))
        limited = self.collector.check()
        deadline = limited["retry_not_before_utc"]
        self.client.get = Mock(wraps=get)
        for elapsed in (60, 600, 899):
            restarted = Collector(self.root, client=self.client, verifier=self.verifier,
                                  now=lambda: now + timedelta(seconds=elapsed))
            deferred = restarted.check()
            self.assertTrue(deferred["network_request_skipped"])
            self.assertEqual(deferred["retry_not_before_utc"], deadline)
            self.assertEqual(deferred["retry_after_seconds"], 900 - elapsed)
            self.assertEqual(deferred["collection_state"], "rate_limited")
            self.assertTrue(deferred["last_good_retained"])
            self.assertEqual((restarted.cache / "latest-success.json").read_bytes(), pointer)
        self.client.get.assert_not_called()
        resumed = Collector(self.root, client=self.client, verifier=self.verifier,
                            now=lambda: now + timedelta(seconds=900)).check()
        self.assertEqual(resumed["status"], "ok")
        self.assertNotIn("retry_not_before_utc", resumed)
        self.client.get.assert_called()

    def test_service_unavailable_cooldown_applies_before_first_snapshot(self):
        now = datetime(2026, 9, 1, 10, 1, tzinfo=timezone.utc)
        self.collector.now = lambda: now
        self.client.get = Mock(side_effect=FetchError("Unavailable", status=503, retry_after=120))
        self.collector.check()
        self.client.get.reset_mock()
        later = Collector(self.root, client=self.client, verifier=self.verifier,
                          now=lambda: now + timedelta(seconds=60)).check()
        self.client.get.assert_not_called()
        self.assertTrue(later["network_request_skipped"])
        self.assertEqual(later["collection_state"], "unavailable")
        self.assertIsNone(later["last_good"])

    def test_cooldown_also_applies_when_switching_feed_environments(self):
        now = datetime(2026, 9, 1, 10, 1, tzinfo=timezone.utc)
        self.collector.now = lambda: now
        self.client.get = Mock(side_effect=FetchError('Wait', status=429, retry_after=900,
                                                    url=BASE_URLS['rehearsal'] + 'index.md5'))
        self.collector.check()
        self.client.get.reset_mock()
        receipt = Collector(self.root, environment='production', client=self.client,
                            now=lambda: now + timedelta(seconds=60)).check()
        self.assertTrue(receipt['network_request_skipped'])
        self.client.get.assert_not_called()

    def test_malformed_latest_check_is_reported_without_network_or_pointer_loss(self):
        self.collector.check()
        pointer = (self.collector.cache / "latest-success.json").read_bytes()
        (self.collector.cache / "latest-check.json").write_text('null')
        self.client.calls.clear()
        failed = self.collector.check()
        self.assertEqual(failed['status'], 'error')
        self.assertIn('valid JSON object', failed['error'])
        self.assertEqual(self.client.calls, [])
        self.assertEqual((self.collector.cache / 'latest-success.json').read_bytes(), pointer)


class HttpTests(unittest.TestCase):
    @staticmethod
    def error(code, retry_after=None):
        headers = Message()
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
        return HTTPError("https://resultat.val.se/test", code, "test", headers, None)

    def test_429_honors_retry_after_and_then_succeeds(self):
        response = io.BytesIO(b"ok")
        opener = Mock(side_effect=[self.error(429, 7), response])
        sleep = Mock()
        self.assertEqual(HttpClient(opener=opener, sleep=sleep).get("https://resultat.val.se/test", 20), b"ok")
        sleep.assert_called_once_with(7)
        self.assertEqual(opener.call_count, 2)

    def test_long_retry_after_defers_without_early_retry(self):
        opener = Mock(side_effect=self.error(429, 3600))
        sleep = Mock()
        with self.assertRaises(FetchError) as raised:
            HttpClient(opener=opener, sleep=sleep).get("https://resultat.val.se/test", 20)
        self.assertEqual(raised.exception.retry_after, 3600)
        opener.assert_called_once()
        sleep.assert_not_called()

    def test_404_reports_unpublished_feed_without_retry(self):
        opener = Mock(side_effect=self.error(404))
        with self.assertRaisesRegex(FetchError, "not yet be published"):
            HttpClient(opener=opener).get("https://resultat.val.se/test", 20)
        opener.assert_called_once()

    def test_missing_resource_url_is_preserved(self):
        for url in (CERTIFICATE_URL, 'https://resultat.val.se/archive.zip'):
            with self.assertRaises(FetchError) as raised:
                HttpClient(opener=Mock(side_effect=self.error(404))).get(url, 1000)
            self.assertEqual(raised.exception.url, url)

    def test_retries_are_finite_and_back_off(self):
        opener, sleep = Mock(side_effect=self.error(503)), Mock()
        with self.assertRaises(FetchError):
            HttpClient(opener=opener, sleep=sleep).get("https://resultat.val.se/test", 20)
        self.assertEqual(opener.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])

    def test_http_date_retry_after(self):
        now = datetime(2026, 9, 13, 18, 0, tzinfo=timezone.utc)
        self.assertEqual(retry_after_seconds("Sun, 13 Sep 2026 18:01:00 GMT", now), 60)

    def test_staleness_uses_swedish_time_for_offsetless_source(self):
        now = datetime(2026, 9, 13, 19, 0, tzinfo=timezone.utc)
        recent = source_freshness("2026-09-13T20:58:00", now, 15)
        self.assertEqual(recent["source_age_seconds"], 120)
        self.assertEqual(recent["source_freshness"], "recent")
        self.assertEqual(source_freshness("2026-09-13T20:00:00", now, 15)["source_freshness"], "stale")

    def test_unsafe_and_duplicate_paths_are_rejected(self):
        checksum = "a" * 32
        for content in (f"{checksum} ../outside.zip", f"{checksum} /outside.zip",
                        f"{checksum} a.zip\n{checksum} a.zip"):
            with self.assertRaises(FeedError):
                parse_index(content.encode())


@unittest.skipUnless(shutil.which("openssl"), "OpenSSL unavailable")
class SignatureTests(unittest.TestCase):
    def test_real_signature_verification_accepts_original_rejects_tampering(self):
        # Ephemeral test key/certificate exercises OpenSSL itself without network.
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            key, cert = folder / "key.pem", folder / "cert.pem"
            data, signature = folder / "data.json", folder / "signature.bin"
            subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                            "-keyout", str(key), "-out", str(cert), "-days", "1",
                            "-subj", "/CN=collector-test"], check=True, capture_output=True)
            data.write_bytes(b'{"test": true}')
            subprocess.run(["openssl", "dgst", "-sha256", "-sign", str(key),
                            "-out", str(signature), str(data)], check=True, capture_output=True)
            members = {"data.json": data.read_bytes(), "data_sign.sha256": signature.read_bytes()}
            verified = verify_signatures(members, cert.read_bytes())
            self.assertEqual(verified["status"], "verified")
            members["data.json"] = b'{"test": false}'
            with self.assertRaises(FeedError):
                verify_signatures(members, cert.read_bytes())


if __name__ == "__main__":
    unittest.main()
