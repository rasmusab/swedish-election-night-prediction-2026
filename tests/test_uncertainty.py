import unittest

import numpy as np
import pandas as pd

from val2026.election_data import PARTIES
from val2026.forecast_model import SwingRegression
from scripts.uncertainty import conditional_share_draws


class ConditionalUncertaintyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(41)
        prior = rng.dirichlet(np.ones(len(PARTIES)) * 3, size=20)
        self.features = pd.DataFrame({
            'code': [str(i) for i in range(20)], 'kind': 'ordinary',
            'municipality': np.repeat(['0001', '0002', '0003', '0004'], 5),
            'county': np.repeat(['00', '01'], 10), 'electorate': 1000,
            'expected_votes': 800., 'prior_rate': .8,
            'baseline_source': 'district_mapping',
            **{'prior_' + p: prior[:, i] for i, p in enumerate(PARTIES)},
        })
        self.observed = np.full((20, len(PARTIES)), np.nan)
        self.observed[:10] = np.array([rng.multinomial(800, p) for p in prior[:10]])
        self.model = SwingRegression(self.features, transform='share')

    def test_unseen_geography_has_finite_nonzero_uncertainty(self):
        result = conditional_share_draws(self.model, self.observed, draws=40, seed=1)
        self.assertEqual(result.diagnostics['unseen_municipalities'], 2)
        self.assertTrue(np.all(result.upper > result.lower))
        np.testing.assert_allclose(result.share_draws.sum(axis=1), 1)
        self.assertGreater(result.diagnostics['residual_df'], 0)

    def test_all_reported_counts_are_fixed(self):
        all_observed = np.tile(self.observed[:10], (2, 1))
        result = conditional_share_draws(self.model, all_observed, draws=20, seed=2)
        expected = all_observed.sum(axis=0) / all_observed.sum()
        np.testing.assert_allclose(result.share_draws, np.tile(expected, (20, 1)))

    def test_partial_report_is_rejected(self):
        bad = self.observed.copy()
        bad[12, 0] = 1
        with self.assertRaises(ValueError):
            conditional_share_draws(self.model, bad, draws=20)


if __name__ == '__main__':
    unittest.main()
