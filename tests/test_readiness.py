"""Preflight must distinguish actual failures, pending production, and offline checks."""
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, Mock
from val2026.readiness import feed_check, run
from val2026.election_feed import BASE_URLS, CERTIFICATE_URL, Collector, FetchError


class ReadinessTests(unittest.TestCase):
    def test_collector_receipt_write_failure_does_not_abort_other_diagnostics(self):
        with TemporaryDirectory() as folder, patch('val2026.readiness.Collector') as collector, \
             patch('val2026.readiness.runtime_check', return_value={}), \
             patch('val2026.readiness.openssl_check', return_value={}), \
             patch('val2026.readiness.certificate_check', return_value={}) as certificate:
            collector.return_value.check.side_effect = OSError('Cannot write receipt')
            report = run(Path(folder))
            self.assertEqual(report['exit_code'], 1)
            self.assertTrue(any(c['name'] == 'Live feed' and 'Cannot write receipt' in c['detail'] for c in report['checks']))
            certificate.assert_called_once()

    def test_production_404_is_pending_before_election_and_failure_after_close(self):
        receipt = {'status': 'error', 'http_status': 404, 'failed_url': BASE_URLS['production'] + 'index.md5'}
        before = datetime(2026, 9, 7, tzinfo=timezone.utc)
        after = datetime(2026, 9, 13, 18, tzinfo=timezone.utc)
        self.assertEqual(feed_check(receipt, 'production', before)[0], 'pending')
        self.assertEqual(feed_check(receipt, 'production', after)[0], 'fail')
        self.assertEqual(feed_check({**receipt, 'failed_url': BASE_URLS['rehearsal'] + 'index.md5'}, 'rehearsal', after)[0], 'warn')

    def test_archive_and_certificate_404_are_failures_even_before_election(self):
        now = datetime(2026, 9, 7, tzinfo=timezone.utc)
        for url in (CERTIFICATE_URL, BASE_URLS['production'] + 'p/rd/Val_2026_preliminar_00_RD.zip'):
            result = feed_check({'status': 'error', 'http_status': 404, 'failed_url': url}, 'production', now)
            self.assertEqual(result[0], 'fail')
            self.assertIn(url, result[1])

    def test_certificate_probe_cooldown_blocks_next_collector_run(self):
        with TemporaryDirectory() as folder, patch('val2026.readiness.Collector') as collector, \
             patch('val2026.readiness.runtime_check', return_value={}), \
             patch('val2026.readiness.inputs_check', return_value={'snapshot_id': 'fixture'}), \
             patch('val2026.readiness.openssl_check', return_value={}), \
             patch('val2026.readiness.certificate_check', side_effect=FetchError('Wait', status=429,
                   retry_after=900, url=CERTIFICATE_URL)):
            root = Path(folder)
            collector.return_value.check.return_value = {'status': 'error', 'http_status': 404,
                                                         'failed_url': BASE_URLS['production'] + 'index.md5'}
            report = run(root, now=datetime(2026, 9, 7, tzinfo=timezone.utc))
            self.assertEqual(report['status'], 'pending')
            client = Mock()
            deferred = Collector(root, environment='production', client=client).check()
            client.get.assert_not_called()
            self.assertTrue(deferred['network_request_skipped'])
            self.assertEqual(deferred['failed_url'], CERTIFICATE_URL)

    def test_network_failure_and_bad_signatures_are_not_unpublished_feed(self):
        now = datetime.now(timezone.utc)
        for error in ('Download timed out', 'Invalid official signature'):
            self.assertEqual(feed_check({'status': 'error', 'error': error}, 'production', now)[0], 'fail')

    def test_offline_never_claims_live_readiness_or_contacts_network(self):
        with TemporaryDirectory() as folder, patch('val2026.readiness.Collector') as collector:
            with patch('val2026.readiness.runtime_check', return_value={}), \
                 patch('val2026.readiness.inputs_check', return_value={'snapshot_id': 'fixture'}), \
                 patch('val2026.readiness.openssl_check', return_value={}):
                report = run(Path(folder), offline=True)
            collector.assert_not_called()
            self.assertEqual(report['status'], 'pending')
            self.assertEqual(report['exit_code'], 2)

    def test_missing_inputs_fail_even_when_production_is_pending(self):
        with TemporaryDirectory() as folder, patch('val2026.readiness.Collector') as collector, \
             patch('val2026.readiness.runtime_check', return_value={}), \
             patch('val2026.readiness.openssl_check', return_value={}), \
             patch('val2026.readiness.certificate_check', return_value={}):
            collector.return_value.check.return_value = {'status': 'error', 'http_status': 404,
                                                         'failed_url': BASE_URLS['production'] + 'index.md5'}
            report = run(Path(folder), now=datetime(2026, 9, 7, tzinfo=timezone.utc))
            self.assertEqual(report['exit_code'], 1)
            self.assertEqual(report['status'], 'needs_attention')
            self.assertTrue(any(c['name'] == 'Frozen inputs and model' and c['status'] == 'fail' for c in report['checks']))
