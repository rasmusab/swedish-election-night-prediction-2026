import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from val2026.election_data import PARTIES

spec = importlib.util.spec_from_file_location('forecast_latest', Path(__file__).resolve().parents[1] / 'scripts/forecast-latest.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def district(code, kind, count=10, reported=True):
    return {'valdistriktskod': code, 'valdistriktstyp': kind, 'kommunkod': '0101',
            'rapporteringsTid': '2026-09-13T22:00:00' if reported else None,
            'rostfordelning': {'rosterPaverkaMandat': {
                'antalRoster': count * 9,
                'partiRoster': [{'partikod': c, 'partiforkortning': p, 'antalRoster': count}
                               for c, p in module.PARTY_CODES.items()],
                'rosterOvrigaPartier': {'antalRoster': count}}}}


class LatestForecastTests(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame({'code': ['01010001', '0101LATE'],
            'kind': ['ordinary', 'late'], 'municipality': ['0101', '0101']})
        self.records = [district('01010001', 'valdistrikt'),
                        district('010100', 'uppsamlingsdistrikt'),
                        district('010101', 'uppsamlingsdistrikt', reported=False)]
        self.roster = [{k: d[k] for k in ['valdistriktskod', 'valdistriktstyp', 'kommunkod']} for d in self.records]

    def test_placeholders_are_missing_not_zero(self):
        self.assertIsNone(module.reporting_counts(district('01010001', 'valdistrikt', count=0, reported=False)))

    def test_partial_late_pool_retains_known_votes_without_claiming_completion(self):
        observed, partial, raw, summary = module.align_observed(self.features, self.records, self.roster)
        self.assertTrue(np.isnan(observed[1]).all())
        np.testing.assert_array_equal(partial[1], np.full(9, 10))
        np.testing.assert_array_equal(raw, np.full(9, 20))
        self.assertEqual(summary['complete_late_municipalities'], 0)

    def test_omitted_late_record_is_still_missing(self):
        observed, partial, raw, summary = module.align_observed(self.features, self.records[:2], self.roster)
        self.assertTrue(np.isnan(observed[1]).all())
        self.assertEqual(summary['incomplete_late_municipalities'][0]['expected_records'], 2)

    def test_complete_late_pool_sums_all_records(self):
        records = self.records[:2] + [district('010101', 'uppsamlingsdistrikt')]
        observed, partial, raw, summary = module.align_observed(self.features, records, self.roster)
        np.testing.assert_array_equal(observed[1], np.full(9, 20))
        self.assertEqual(summary['complete_late_municipalities'], 1)

    def test_partial_lower_bound_preserves_complete_counts(self):
        observed, partial, _, _ = module.align_observed(self.features, self.records, self.roster)
        predicted = np.full((2, 9), 5.)
        predicted[0] = observed[0]
        bounded, _ = module.preserve_partial_lower_bounds(predicted, observed, partial)
        np.testing.assert_array_equal(bounded[0], observed[0])
        self.assertTrue((bounded[1] >= partial[1]).all())


if __name__ == '__main__':
    unittest.main()
