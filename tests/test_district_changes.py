"""Changed-district scoring excludes observed returns and pools all municipal fallbacks."""
import importlib.util
import unittest
import numpy as np
import pandas as pd
from val2026 import ROOT
from val2026.election_data import PARTIES

spec = importlib.util.spec_from_file_location('district_changes', ROOT / 'scripts/check-district-changes.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DistrictChangeTests(unittest.TestCase):
    def test_weighted_fallback_score_includes_average_and_excludes_reported_and_late(self):
        features = pd.DataFrame({'kind': ['ordinary'] * 3 + ['late'],
            'baseline_source': ['municipality_residual', 'municipality_average', 'municipality_residual', 'municipality_late'],
            'electorate': [100, 300, 10000, 10000]})
        final = np.zeros((4, 9)); final[:, PARTIES.index('M')] = [50, 150, 5000, 5000]
        final[:, PARTIES.index('S')] = [50, 150, 5000, 5000]
        predicted = final.copy()
        predicted[:2, PARTIES.index('M')] = [60, 240]
        predicted[:2, PARTIES.index('S')] = [40, 60]
        predicted[2:, PARTIES.index('M')] = 10000
        predicted[2:, PARTIES.index('S')] = 0
        observed = np.full_like(final, np.nan); observed[2] = final[2]
        rows, _ = module.district_scores(features, observed, predicted, final)
        fallback = next(r for r in rows if r['baseline_source'] == 'municipal_fallback')
        self.assertEqual(fallback['districts'], 2)
        self.assertEqual(fallback['final_votes'], 400)
        self.assertAlmostEqual(fallback['district_share_mae_pp'], 6.25)
        self.assertAlmostEqual(fallback['aggregate_share_mae_pp'], 6.25)
        self.assertEqual(fallback['volume_wape_pct'], 0)

    def test_other_stays_in_denominator_and_empty_groups_are_not_perfect_scores(self):
        features = pd.DataFrame({'kind': ['ordinary'], 'baseline_source': ['district_mapping'], 'electorate': [100]})
        final = np.zeros((1, 9)); final[0, 0] = 50; final[0, -1] = 50
        predicted = final.copy(); predicted[0, 0] = 60; predicted[0, -1] = 40
        rows, _ = module.district_scores(features, np.full_like(final, np.nan), predicted, final)
        self.assertAlmostEqual(rows[0]['district_share_mae_pp'], 1.25)
        self.assertTrue(np.isnan(rows[1]['district_share_mae_pp']))
        rows, _ = module.district_scores(features, final, predicted, final)
        self.assertTrue(all(r['districts'] == 0 and np.isnan(r['district_share_mae_pp']) for r in rows))

    def test_exposure_uses_electorate_and_includes_municipality_average(self):
        features = pd.DataFrame({'code': ['a', 'b', 'c'], 'kind': ['ordinary'] * 3,
            'baseline_source': ['district_mapping', 'municipality_average', 'municipality_residual'],
            'constituency': ['01'] * 3, 'electorate': [200, 50, 50]})
        exposure = module.exposure_table(features).iloc[0]
        self.assertEqual(exposure.fallback_districts, 2)
        self.assertAlmostEqual(exposure.fallback_electorate_fraction, 1/3)
        self.assertEqual(exposure.exposure_group, 'low')
