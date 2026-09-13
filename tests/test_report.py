from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from val2026.election_feed import BASE_URLS, CERTIFICATE_URL, FetchError
from val2026.election_forecast import load_inputs, resolve_roster, reporting_counts
from val2026.election_report import update_lock, phase_at, run_cycle
from scripts.rehearsal_scenarios import Scenario, partial

class ReportWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scenario = Scenario(self.root)

    def test_only_unpublished_production_index_is_waiting_before_polls_close(self):
        from unittest.mock import Mock
        now = datetime(2026, 9, 8, 8, tzinfo=timezone.utc)
        index = BASE_URLS['production'] + 'index.md5'
        for url, health in ((index, 'waiting'), (CERTIFICATE_URL, 'update_failed'),
                            (BASE_URLS['production'] + 'p/rd/result.zip', 'update_failed')):
            with self.subTest(url=url):
                collector = Mock()
                collector.check.return_value = {'status': 'error', 'http_status': 404,
                                                'failed_url': url, 'error': 'HTTP 404'}
                state = run_cycle(self.root, 'production', now=now, collector=collector)
                self.assertEqual(state['health'], health)

    def test_sequence_missing_early_correction_failure_and_recovery(self):
        s = self.scenario
        waiting = s.run(partial(s.source,0))
        self.assertEqual(waiting['estimate_state'],'waiting')
        self.assertNotIn('forecast_share_pct',waiting['parties'][0])
        self.assertNotIn('seat_allocation', waiting)
        early = s.run(partial(s.source,100,step=1))
        self.assertEqual(early['estimate_state'],'early')
        self.assertEqual(early['training_rows'],100)
        data = partial(s.source,1000,step=2)
        forecast = s.run(data)
        self.assertEqual(forecast['estimate_state'],'forecast')
        self.assertEqual(sum(p['forecast_seats'] for p in forecast['parties']), 349)
        self.assertEqual(sum(p['fixed_seats'] for p in forecast['parties']), 310)
        self.assertEqual(sum(p['adjustment_seats'] for p in forecast['parties']), 39)
        self.assertEqual([b['parties'] for b in forecast['blocs']],
                         [['V', 'S', 'MP', 'C'], ['M', 'L', 'KD', 'SD']])
        self.assertEqual(sum(b['seats'] for b in forecast['blocs']), 349)
        for bloc in forecast['blocs']:
            self.assertEqual(bloc['seats'], sum(p['forecast_seats'] for p in forecast['parties']
                                              if p['party'] in bloc['parties']))
        self.assertEqual(forecast['history'][-1]['seats'], forecast['seat_allocation']['national_seats'])
        uncertainty = forecast['uncertainty']
        self.assertEqual(uncertainty['draws'], 32)
        self.assertTrue(all(sum(draw) == 349 for draw in uncertainty['seat_draws']))
        self.assertEqual(forecast['history'][-1]['bloc_intervals']['V + S + MP + C'],
                         forecast['blocs'][0]['uncertainty'])
        for party in uncertainty['parties']:
            interval = party['seats']
            self.assertLessEqual(interval['lower_90'], interval['lower_50'])
            self.assertLessEqual(interval['upper_50'], interval['upper_90'])
        counted = forecast['raw_valid_votes']
        revised = deepcopy(data)
        revised['senasteUppdateringstid'] = '2026-09-13T20:03:00+00:00'
        valid = next(d for d in revised['valdistrikt'] if d['rapporteringsTid'])['rostfordelning']['rosterPaverkaMandat']
        valid['partiRoster'][0]['antalRoster'] -= 1
        valid['antalRoster'] -= 1
        corrected = s.run(revised)
        self.assertEqual(corrected['raw_valid_votes'],counted-1)
        history_len = len(corrected['history'])
        unchanged = s.run()
        self.assertEqual(len(unchanged['history']),history_len)
        estimate_time = unchanged['estimate_generated_at_utc']
        s.client.error = FetchError('Synthetic outage',status=503)
        failed = s.run()
        self.assertEqual(failed['health'],'update_failed')
        self.assertEqual(failed['estimate_generated_at_utc'],estimate_time)
        self.assertEqual(failed['parties'],unchanged['parties'])
        self.assertEqual(failed['blocs'], unchanged['blocs'])
        self.assertEqual(failed['seat_allocation'], unchanged['seat_allocation'])
        self.assertEqual(failed['uncertainty'], unchanged['uncertainty'])
        s.client.error = None
        recovered = s.run()
        self.assertNotEqual(recovered['health'],'update_failed')

    def test_incomplete_roster_never_completes_late_pools(self):
        s = self.scenario
        data = partial(s.source,10,1,omit_unreported=True)
        state = s.run(data)
        self.assertEqual(state['roster_source']['status'],'pending_full_feed')
        self.assertEqual(state['complete_late_municipalities'],0)
        self.assertFalse((s.collector.cache / 'roster.json').exists())
        complete = s.run(partial(s.source,10,1,step=1))
        self.assertEqual(complete['roster_source']['status'],'validated_full_feed')
        self.assertTrue((s.collector.cache / 'roster.json').exists())

    def test_missing_party_fails_and_retains_estimate(self):
        s = self.scenario
        good = s.run(partial(s.source,10))
        malformed = partial(s.source,11,step=1)
        next(d for d in malformed['valdistrikt'] if d['rapporteringsTid'])['rostfordelning']['rosterPaverkaMandat']['partiRoster'].pop()
        failed = s.run(malformed)
        self.assertEqual(failed['health'],'update_failed')
        self.assertEqual(failed['parties'],good['parties'])

    def test_changed_or_tampered_inputs_do_not_overwrite_estimate(self):
        s = self.scenario
        good = s.run(partial(s.source,10))
        manifest = json.loads((self.root/'data/inputs/latest-success.json').read_bytes())
        with (self.root/manifest['baseline_path']).open('ab') as f:
            f.write(b'changed')
        failed = s.run()
        self.assertEqual(failed['health'],'update_failed')
        self.assertEqual(failed['parties'],good['parties'])

    def test_production_roster_uses_no_rehearsal_archive(self):
        features, _ = load_inputs(self.root)
        data = partial(self.scenario.source,0)
        data.pop('test')
        roster, complete, _ = resolve_roster(self.root,'production',features,data,{'archive_sha256':'fixture'})
        self.assertTrue(complete)
        self.assertEqual(len(roster),len(data['valdistrikt']))
        self.assertFalse((self.root/'data/Genrep_2026_preliminar_00_RD.zip').exists())

    def test_constituency_change_fails_before_replacing_success(self):
        s = self.scenario
        good = s.run(partial(s.source, 10))
        data = partial(s.source, 11, step=1)
        record = data['valdistrikt'][0]
        record['kretskod'] = '29' if record['kretskod'] != '29' else '01'
        failed = s.run(data)
        self.assertEqual(failed['health'], 'update_failed')
        self.assertIn('constituency', failed['error'])
        self.assertEqual(failed['seat_allocation'], good['seat_allocation'])

    def test_fixed_seat_tampering_retains_previous_seats(self):
        s = self.scenario
        good = s.run(partial(s.source, 10))
        manifest = json.loads((self.root / 'data/inputs/latest-success.json').read_bytes())
        with (self.root / manifest['fixed_seats_path']).open('ab') as handle:
            handle.write(b'changed')
        failed = s.run()
        self.assertEqual(failed['health'], 'update_failed')
        self.assertIn('Frozen input changed', failed['error'])
        self.assertEqual(failed['blocs'], good['blocs'])

    def test_uncertainty_failure_preserves_complete_previous_forecast(self):
        from unittest.mock import patch
        s = self.scenario
        good = s.run(partial(s.source, 10))
        with patch('val2026.forecast_uncertainty.forecast_uncertainty', side_effect=ValueError('Injected simulation failure')):
            failed = s.run(partial(s.source, 20, step=1))
        self.assertEqual(failed['health'], 'update_failed')
        self.assertEqual(failed['uncertainty'], good['uncertainty'])
        self.assertEqual(failed['estimate_generated_at_utc'], good['estimate_generated_at_utc'])
        self.assertEqual(failed['history'], good['history'])

    def test_uncertainty_code_tampering_fails_the_frozen_lock(self):
        s = self.scenario
        good = s.run(partial(s.source, 10))
        with (self.root / 'val2026/uncertainty.py').open('ab') as handle:
            handle.write(b'changed')
        failed = s.run()
        self.assertEqual(failed['health'], 'update_failed')
        self.assertEqual(failed['uncertainty'], good['uncertainty'])

    def test_lock_prevents_overlapping_updates(self):
        with update_lock(self.root):
            with self.assertRaises(RuntimeError):
                with update_lock(self.root):
                    self.fail('acquired second lock')

    def test_phase_boundaries_use_swedish_time(self):
        self.assertEqual(phase_at(datetime(2026,9,13,17,59,tzinfo=timezone.utc)),'before_polls_close')
        self.assertEqual(phase_at(datetime(2026,9,13,18,tzinfo=timezone.utc)),'election_night')
        self.assertEqual(phase_at(datetime(2026,9,14,10,tzinfo=timezone.utc)),'preliminary_pause')
        self.assertEqual(phase_at(datetime(2026,9,16,10,tzinfo=timezone.utc)),'late_preliminary_count')

if __name__ == '__main__':
    unittest.main()

class HtmlExportTests(unittest.TestCase):
    def test_only_default_production_render_updates_homepage(self):
        from unittest.mock import patch
        from val2026.election_forecast import ROOT
        spec = importlib.util.spec_from_file_location('update_report_pages_test', ROOT/'update-report.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        html = '<html><body>Election-night estimate</body></html>'
        for environment, custom in [('production', False), ('rehearsal', False), ('production', True)]:
            with self.subTest(environment=environment, custom=custom), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                homepage = root/'index.html'
                homepage.write_text('previous homepage')
                state_path = root/'outputs'/f'latest-{environment}'/'report-state.json'
                state_path.parent.mkdir(parents=True)
                state_path.write_text(json.dumps({'test': environment == 'rehearsal', 'health': 'waiting'}))
                output = root/'outputs/custom/index.html' if custom else None
                with patch.object(module.NotebookClient, 'execute'), patch.object(
                        module.HTMLExporter, 'from_notebook_node', return_value=(html, {})):
                    receipt = module.render(root, environment, output=output)
                expected = html if environment == 'production' and not custom else 'previous homepage'
                self.assertEqual(homepage.read_text(), expected)
                self.assertEqual(Path(receipt['output']).read_text(), html)
                self.assertFalse((root/'executed.ipynb').exists())
                self.assertFalse((root/'render-receipt.json').exists())

    def test_failed_notebook_keeps_previous_html(self):
        from unittest.mock import patch
        from val2026.election_forecast import ROOT
        spec = importlib.util.spec_from_file_location('update_report_test', ROOT/'update-report.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root/'index.html'
            output.write_text('previous successful page')
            with patch.object(module.NotebookClient,'execute',side_effect=RuntimeError('injected execution failure')):
                with self.assertRaises(RuntimeError):
                    module.render(root,'rehearsal',output=output)
            self.assertEqual(output.read_text(),'previous successful page')
