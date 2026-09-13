from datetime import datetime, timedelta, timezone
from email.message import Message
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError

from val2026 import ROOT
from val2026.election_feed import FetchError, RetryDeferred, record_retry_deadline

spec = importlib.util.spec_from_file_location('operational_rehearsal_test', ROOT / 'scripts/operational-rehearsal.py')
drill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drill)


class OperationalRehearsalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.shared = self.root / 'shared'
        self.workspace = self.root / 'workspace'

    def test_fresh_cache_does_not_bypass_shared_cooldown(self):
        deadline = datetime.now(timezone.utc) + timedelta(minutes=10)
        record_retry_deadline(self.shared, RetryDeferred(deadline, now=datetime.now(timezone.utc), status=429))
        opener = Mock()
        client = drill.SharedCooldownClient(self.shared, opener=opener, max_attempts=1)
        with self.assertRaises(RetryDeferred):
            client.get('https://example.invalid/result', 100)
        opener.assert_not_called()
        drill.share_cooldown(self.shared, self.workspace)
        self.assertEqual(drill.read_cooldown(self.workspace).deadline, deadline)

    def test_live_failure_records_the_deadline_in_the_shared_project(self):
        headers = Message()
        headers['Retry-After'] = '120'
        opener = Mock(side_effect=HTTPError('https://example.invalid/result', 429, 'Limited', headers, None))
        client = drill.SharedCooldownClient(self.shared, opener=opener, max_attempts=1)
        before = datetime.now(timezone.utc)
        with self.assertRaises(FetchError):
            client.get('https://example.invalid/result', 100)
        deadline = drill.read_cooldown(self.shared).deadline
        self.assertGreaterEqual(deadline, before + timedelta(seconds=120))
        self.assertLess(deadline, before + timedelta(seconds=125))

    def test_merging_deadlines_neither_shortens_nor_restarts_them(self):
        now = datetime.now(timezone.utc)
        first, later = now + timedelta(minutes=5), now + timedelta(minutes=10)
        record_retry_deadline(self.shared, RetryDeferred(later, now=now, status=429))
        record_retry_deadline(self.workspace, RetryDeferred(first, now=now, status=429))
        drill.share_cooldown(self.workspace, self.shared)
        self.assertEqual(drill.read_cooldown(self.shared).deadline, later)
        drill.share_cooldown(self.shared, self.workspace)
        self.assertEqual(drill.read_cooldown(self.workspace).deadline, later)

    def test_disconnection_uses_real_http_error_handling_without_network_changes(self):
        result = drill.check_feed(self.workspace, self.shared, disconnected=True)
        self.assertEqual(result['error_kind'], 'download_error')
        self.assertIn('Injected operational rehearsal disconnection', result['error'])
        self.assertFalse(result.get('network_request_skipped', False))
