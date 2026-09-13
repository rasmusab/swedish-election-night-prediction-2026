import unittest

import numpy as np
import pandas as pd

from val2026.election_data import PARTIES
from val2026.forecast_model import SwingRegression
from val2026.uncertainty import simulate_constituency_votes
from val2026.forecast_uncertainty import summarize_vote_draws


class PredictiveUncertaintyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(41)
        n = 24
        prior = rng.dirichlet(np.ones(len(PARTIES)) * 4, size=n)
        self.features = pd.DataFrame({
            'code': [str(i) for i in range(n)], 'kind': ['ordinary'] * 20 + ['late'] * 4,
            'municipality': list(np.repeat(['0001', '0002', '0003', '0004'], 5)) + ['0001', '0002', '0003', '0004'],
            'county': list(np.repeat(['01', '02'], 10)) + ['01', '01', '02', '02'],
            'constituency': list(np.repeat(['01', '02'], 10)) + ['01', '01', '02', '02'],
            'electorate': 1000., 'expected_votes': [800.] * 20 + [80.] * 4,
            'prior_rate': .8, 'baseline_source': 'district_mapping',
            **{'prior_' + p: prior[:, i] for i, p in enumerate(PARTIES)}})
        self.truth = np.array([rng.multinomial(int(v), p) for v, p in zip(self.features.expected_votes, prior)])
        self.observed = np.full((n, len(PARTIES)), np.nan)
        self.observed[:10] = self.truth[:10]
        self.model = SwingRegression(self.features, transform='share', early_guard=True)

    def simulate(self, observed=None, **kwargs):
        return simulate_constituency_votes(self.model, self.observed if observed is None else observed,
                                           draws=64, seed=1, **kwargs)

    def test_finite_joint_votes_reproducible_and_point_input_unchanged(self):
        predicted = self.model.fit_predict(self.observed)
        original = predicted.copy()
        first = self.simulate(predicted=predicted)
        second = self.simulate(predicted=predicted)
        np.testing.assert_array_equal(first.vote_draws, second.vote_draws)
        np.testing.assert_array_equal(predicted, original)
        self.assertEqual(first.vote_draws.shape, (64, 2, 9))
        self.assertTrue(np.isfinite(first.vote_draws).all())
        self.assertTrue((first.vote_draws >= 0).all())

    def test_complete_preliminary_counts_exact_but_final_revision_uncertainty_remains(self):
        result = self.simulate(self.truth)
        expected = self.truth.reshape(24, 9)
        grouped = np.array([expected[self.features.constituency.eq(c)].sum(axis=0) for c in result.constituency_codes])
        np.testing.assert_allclose(result.preliminary_vote_draws, np.broadcast_to(grouped, (64, 2, 9)))
        self.assertGreater(result.vote_draws[:, 0, 0].std(), 0)
        no_revision = self.simulate(self.truth, config={'final_revisions': False})
        np.testing.assert_array_equal(no_revision.vote_draws, no_revision.preliminary_vote_draws)

    def test_reported_counts_can_exceed_frozen_electorate_without_being_changed(self):
        truth = self.truth.copy()
        truth[0] *= 2
        result = self.simulate(truth, config={'final_revisions': False})
        expected = truth[self.features.constituency.eq('01')].sum(axis=0)
        np.testing.assert_allclose(result.preliminary_vote_draws[:, 0], np.tile(expected, (64, 1)))

    def test_partial_late_votes_remain_a_floor_before_revisions(self):
        partial = np.zeros_like(self.observed)
        partial[20] = 30
        result = self.simulate(partial_late=partial, config={'final_revisions': False})
        floor = self.truth[:10].sum(axis=0) + partial[20]
        self.assertTrue((result.preliminary_vote_draws[:, 0] >= floor - 1e-7).all())

    def test_tiny_reported_sample_has_prior_variance_floor(self):
        observed = np.full_like(self.observed, np.nan)
        observed[:2] = self.truth[:2]
        result = self.simulate(observed)
        self.assertGreater(result.diagnostics['residual_prior_weight'], .9)
        self.assertGreater(result.vote_draws.sum(axis=1)[:, 0].std(), 1)

    def test_incomplete_late_pool_remains_uncertain_when_floor_exceeds_point(self):
        observed = self.truth.astype(float).copy()
        observed[20] = np.nan
        partial = np.zeros_like(observed)
        partial[20] = 100
        result = self.simulate(observed, partial_late=partial,
                               config={'final_revisions': False, 'coefficient_uncertainty': False,
                                       'district_residuals': False, 'late_effect': False})
        self.assertEqual(result.diagnostics['partial_late_pools_with_one_sided_remainder'], 1)
        self.assertGreater(result.preliminary_vote_draws[:, 0].sum(axis=1).std(), 1)
        floor = observed[self.features.constituency.eq('01') & np.isfinite(observed).all(axis=1)].sum(axis=0) + partial[20]
        self.assertTrue((result.preliminary_vote_draws[:, 0] >= floor).all())

    def test_zero_components_reproduce_point_without_recentring(self):
        config = {k: False for k in ['coefficient_uncertainty', 'district_residuals',
                                    'reporting_discrepancy', 'late_effect', 'late_residuals', 'final_revisions']}
        predicted = self.model.fit_predict(self.observed)
        result = self.simulate(predicted=predicted, config=config)
        expected = np.array([predicted[self.features.constituency.eq(c)].sum(axis=0) for c in result.constituency_codes])
        np.testing.assert_allclose(result.vote_draws, np.broadcast_to(expected, result.vote_draws.shape))

    def test_invalid_partial_or_incomplete_counts_are_rejected(self):
        partial = np.zeros_like(self.observed)
        partial[1, 0] = 2
        with self.assertRaises(ValueError):
            self.simulate(partial_late=partial)
        observed = self.observed.copy()
        observed[11, 0] = 1
        with self.assertRaises(ValueError):
            self.simulate(observed)

    def test_joint_seats_preserve_threshold_jump_and_bloc_complements(self):
        votes = np.zeros((4, 1, 9))
        votes[:, 0, PARTIES.index('S')] = 50
        votes[:, 0, PARTIES.index('L')] = [3.9, 4.1, 3.8, 4.2]
        votes[:, 0, PARTIES.index('M')] = 50 - votes[:, 0, PARTIES.index('L')]
        result = summarize_vote_draws(votes, ['01'], {'01': 310})
        liberal = next(p for p in result['parties'] if p['party'] == 'L')
        self.assertEqual(liberal['national_threshold_probability'], .5)
        self.assertEqual(liberal['seats']['lower_90'], 0)
        self.assertGreater(liberal['seats']['upper_90'], 10)
        self.assertTrue(np.all(np.array(result['seat_draws']).sum(axis=1) == 349))
        self.assertEqual(sum(b['majority_probability'] for b in result['blocs']), 1)
        for bloc in result['blocs']:
            totals = np.array(result['seat_draws'])[:, [PARTIES.index(p) for p in bloc['parties']]].sum(axis=1)
            self.assertEqual(bloc['majority_probability'], np.count_nonzero(totals >= 175) / len(totals))


if __name__ == '__main__':
    unittest.main()
