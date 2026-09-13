"""An invalid upstream refresh must not replace the active election baseline."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import pandas as pd
from val2026.election_forecast import ROOT

spec = importlib.util.spec_from_file_location('refresh_inputs_test', ROOT/'refresh-inputs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FrozenInputTests(unittest.TestCase):
    def test_invalid_download_preserves_active_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pointer = root/'data/inputs/latest-success.json'
            pointer.parent.mkdir(parents=True)
            original = b'{"snapshot_id":"previously-validated"}\n'
            pointer.write_bytes(original)
            client = Mock()
            client.get.return_value = b'Temporary upstream error page, not a workbook'
            with patch.object(module,'HttpClient',return_value=client), patch.object(module,'official_urls',return_value={n:'https://www.val.se/fixture/'+n for n in module.SOURCES}):
                with self.assertRaises(ValueError):
                    module.refresh(root)
            self.assertEqual(pointer.read_bytes(),original)
            self.assertFalse((root/'data/processed/baseline-2026.csv').exists())

    def test_current_geography_and_fixed_seats_are_frozen_together(self):
        baseline, roster, seats, validation = module.prepare(ROOT/'data')
        self.assertEqual(len(seats), 29)
        self.assertEqual(seats.fixed_seats.sum(), 310)
        self.assertEqual(seats.set_index('constituency').loc['02', 'fixed_seats'], 41)
        self.assertEqual(seats.set_index('constituency').loc['09', 'fixed_seats'], 2)
        self.assertEqual(set(baseline.constituency), set(seats.constituency))
        self.assertFalse(baseline.constituency.isna().any())
        self.assertEqual(len(baseline[baseline.kind.eq('late')]), 290)
        self.assertTrue(baseline.groupby('municipality').constituency.nunique().eq(1).all())
        self.assertEqual(set(roster.constituency), set(seats.constituency))
        self.assertEqual(validation['fixed_seats'], 310)
        # County 01 contains two Riksdag constituencies; county codes cannot
        # substitute for the identifiers used by the allocation algorithm.
        self.assertEqual(set(roster.loc[roster.county.eq('01'), 'constituency']), {'01', '02'})

    def test_rejects_municipality_split_between_constituencies(self):
        roster = pd.DataFrame({'municipality': [f'{i:04d}' for i in range(1, 30)],
                               'constituency': [f'{i:02d}' for i in range(1, 30)],
                               'constituency_name': [f'Constituency {i}' for i in range(1, 30)]})
        ambiguous = roster.iloc[[0]].copy()
        ambiguous['constituency'] = '02'
        ambiguous['constituency_name'] = 'Constituency 2'
        with self.assertRaisesRegex(ValueError, 'Municipality late-vote pool'):
            module.validate_constituencies(pd.concat([roster, ambiguous], ignore_index=True))

    def test_rejects_inconsistent_fixed_seat_decision(self):
        names = [f'Constituency {i}' for i in range(1, 30)]
        ordinary = pd.DataFrame({'constituency': [f'{i:02d}' for i in range(1, 30)],
                                 'constituency_name': names})
        # Every row has the correct parliamentary total, but the fixed seat
        # sum is wrong: the separate 310-seat validation must catch this.
        upstream = pd.DataFrame({'Valtyp': ['Val till Riksdagen']*29,
                                 'Valkrets': names, 'Fasta mandat': [10]*29,
                                 'Totalt antal mandat': [349]*29})
        with patch.object(module.pd, 'read_excel', return_value=upstream):
            with self.assertRaisesRegex(ValueError, 'summing to 310'):
                module.fixed_seats_2026(Path('fixture.xlsx'), ordinary)

    def test_active_manifest_has_hashed_seat_input(self):
        manifest = json.loads((ROOT/'data/inputs/latest-success.json').read_bytes())
        payload = (ROOT/manifest['fixed_seats_path']).read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), manifest['fixed_seats_sha256'])
        self.assertEqual(manifest['validation']['riksdag_constituencies'], 29)

if __name__ == '__main__':
    unittest.main()
