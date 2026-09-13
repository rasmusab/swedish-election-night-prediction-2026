from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts.rehearsal_scenarios import (Scenario, aggregate_documents, partial, template,
                                         MEMBER, MANDATE_MEMBER)
from val2026.election_forecast import ROOT, load_inputs, load_snapshot
from val2026.reconciliation import reconcile_aggregates


class AggregateReconciliationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roster = template()
        features, _ = load_inputs(ROOT)
        cls.geography = features.groupby('municipality').constituency.first().to_dict()

    def setUp(self):
        self.source = partial(self.roster, 25, 3, omit_unreported=True)
        self.docs = aggregate_documents(self.source, self.roster)

    def reconcile(self, docs=None, environment='rehearsal'):
        docs = self.docs if docs is None else docs
        return reconcile_aggregates(docs['rostfordelning'], docs['mandatfordelning'],
                                    docs['summering'], environment, self.geography)

    def test_partial_roster_and_partial_late_returns_reconcile(self):
        result = self.reconcile()
        self.assertEqual(result['reported_districts'], 28)
        self.assertFalse(result['district_roster_complete'])
        self.assertEqual((result['constituencies_checked'], result['municipalities_checked']), (29, 290))
        expected = sum(d['rostfordelning']['rosterPaverkaMandat']['antalRoster'] for d in self.source['valdistrikt'])
        self.assertEqual(result['reported_valid_votes'], expected)

    def test_file_generation_timestamps_and_counters_need_not_be_identical(self):
        self.docs['mandatfordelning']['senasteUppdateringstid'] = '2026-09-13T20:00:02+00:00'
        self.docs['summering']['senasteUppdateringstid'] = '2026-09-13T20:00:03+00:00'
        self.docs['summering']['antalUppdateringar'] += 1
        result = self.reconcile()
        self.assertEqual(result['status'], 'reconciled')
        self.assertNotEqual(result['source_updated_at']['districts'], result['source_updated_at']['municipalities'])

    def test_each_aggregation_level_independently_detects_party_vote_mismatch(self):
        for level in ('national', 'constituency', 'municipality'):
            with self.subTest(level=level):
                docs = deepcopy(self.docs)
                national = docs['mandatfordelning']['valomrade']
                areas = ([national] if level == 'national' else national['valkretsLista'] if level == 'constituency'
                         else docs['summering']['kommuner'])
                area = next(a for a in areas if a['antalValdistriktRaknade'])
                valid = area['rostfordelning']['rosterPaverkaMandat']
                valid['partiRoster'][0]['antalRoster'] += 1
                valid['antalRoster'] += 1
                with self.assertRaisesRegex(ValueError, level + '.*party votes differ'):
                    self.reconcile(docs)

    def test_reported_aggregate_requires_every_party_even_if_other_counts_match(self):
        valid = self.docs['mandatfordelning']['valomrade']['rostfordelning']['rosterPaverkaMandat']
        valid['partiRoster'].pop()
        with self.assertRaisesRegex(ValueError, 'lacks explicit counts'):
            self.reconcile()

    def test_unreported_areas_allow_null_and_zero_placeholders_but_never_votes(self):
        area = next(a for a in self.docs['mandatfordelning']['valomrade']['valkretsLista']
                    if a['antalValdistriktRaknade'] == 0)
        area['rostfordelning'] = {'rosterPaverkaMandat': {
            'antalRoster': 0, 'partiRoster': [{'partikod': '0001', 'antalRoster': 0}]}}
        self.assertEqual(self.reconcile()['status'], 'reconciled')
        area['rostfordelning']['rosterPaverkaMandat']['partiRoster'][0]['antalRoster'] = 1
        with self.assertRaisesRegex(ValueError, 'votes but no reported districts'):
            self.reconcile()

    def test_reported_area_cannot_hide_missing_counts_behind_a_null_distribution(self):
        self.docs['mandatfordelning']['valomrade']['rostfordelning'] = None
        with self.assertRaisesRegex(ValueError, 'lacks explicit reported party votes'):
            self.reconcile()

    def test_counted_district_mismatch_is_rejected(self):
        self.docs['mandatfordelning']['valomrade']['antalValdistriktRaknade'] += 1
        with self.assertRaisesRegex(ValueError, 'reported district count differs'):
            self.reconcile()

    def test_partial_local_expected_counts_must_sum_to_national_count(self):
        self.docs['summering']['kommuner'][0]['antalValdistriktSomSkaRaknas'] += 1
        with self.assertRaisesRegex(ValueError, 'expected counts do not sum'):
            self.reconcile()

    def test_missing_or_duplicate_areas_are_rejected(self):
        for kind in ('missing', 'duplicate'):
            with self.subTest(kind=kind):
                docs = deepcopy(self.docs)
                if kind == 'missing':
                    docs['mandatfordelning']['valomrade']['valkretsLista'].pop()
                else:
                    docs['summering']['kommuner'].append(deepcopy(docs['summering']['kommuner'][0]))
                with self.assertRaisesRegex(ValueError, 'coverage differs|duplicate municipalities'):
                    self.reconcile(docs)

    def test_election_identity_and_environment_must_match_across_members(self):
        self.docs['mandatfordelning']['valtillfalle'] = 'Another election'
        with self.assertRaisesRegex(ValueError, 'election metadata differs'):
            self.reconcile()
        self.docs['mandatfordelning']['valtillfalle'] = self.source['valtillfalle']
        for document in self.docs.values():
            document.pop('test')
        self.assertEqual(self.reconcile(environment='production')['status'], 'reconciled')
        self.docs['summering']['test'] = True
        with self.assertRaisesRegex(Exception, 'Wrong test flag'):
            self.reconcile(environment='production')

    def test_real_signature_verified_rehearsal_reconciles_all_areas(self):
        receipt, _ = load_snapshot(ROOT, 'rehearsal')
        check = receipt['aggregate_reconciliation']
        self.assertEqual(check['reported_valid_votes'], 6_877_640)
        self.assertEqual(check['reported_districts'], 6_626)
        self.assertEqual(len(check['signed_archive_members']), 3)


class AggregateWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scenario = Scenario(self.root)

    def test_inconsistent_archive_retains_complete_forecast_then_recovers(self):
        scenario = self.scenario
        good = scenario.run(partial(scenario.source, 10))
        data = partial(scenario.source, 11, step=1)
        docs = aggregate_documents(data, scenario.source)
        valid = docs['mandatfordelning']['valomrade']['rostfordelning']['rosterPaverkaMandat']
        valid['partiRoster'][0]['antalRoster'] += 1
        valid['antalRoster'] += 1
        scenario.client.set(data, documents=docs)
        failed = scenario.run()
        self.assertEqual(failed['health'], 'update_failed')
        self.assertIn('Aggregate reconciliation', failed['error'])
        for key in ('parties', 'seat_allocation', 'uncertainty', 'estimate_generated_at_utc', 'history'):
            self.assertEqual(failed[key], good[key])
        recovered = scenario.run(data)
        self.assertNotEqual(recovered['health'], 'update_failed')
        self.assertEqual(recovered['aggregate_reconciliation']['reported_districts'], 11)

    def test_missing_aggregate_or_missing_signature_provenance_is_rejected(self):
        scenario = self.scenario
        data = partial(scenario.source, 10)
        docs = aggregate_documents(data, scenario.source)
        docs.pop('summering')
        scenario.client.set(data, documents=docs)
        missing = scenario.run()
        self.assertEqual(missing['health'], 'update_failed')
        self.assertIn('signature-verified summering', missing['error'])
        scenario.collector.verifier = lambda members, certificate: {
            'status': 'verified', 'json_files_verified': [MEMBER, MANDATE_MEMBER], 'fixture': True}
        unsigned = scenario.run(data)
        self.assertEqual(unsigned['health'], 'update_failed')
        self.assertIn('signature-verified summering', unsigned['error'])
