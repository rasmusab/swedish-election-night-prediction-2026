"""Checks that future returns cannot leak into the replay predictor."""
import unittest
import numpy as np
from val2026.replay import load_backtest, arrival_order, reveal, simple_forecast
from val2026.forecast_model import SwingRegression


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features, cls.prelim, cls.final = load_backtest()

    def test_no_outcomes_in_features(self):
        self.assertFalse(any(c.startswith(('prelim_', 'final_')) for c in self.features))

    def test_future_returns_do_not_change_prediction(self):
        order = arrival_order(self.features, 'small_first', 123)
        seen = reveal(self.prelim, order, 0.1)
        future_changed = self.prelim.copy()
        future_changed[~np.isfinite(seen).all(axis=1)] *= 100
        revealed_again = reveal(future_changed, order, 0.1)
        np.testing.assert_equal(seen, revealed_again)
        np.testing.assert_allclose(simple_forecast(self.features, seen, 'county_swing'),
                                   simple_forecast(self.features, revealed_again, 'county_swing'))

    def test_late_votes_never_revealed_on_election_night(self):
        order = arrival_order(self.features, 'random', 123)
        seen = reveal(self.prelim, order, 1.0)
        self.assertTrue(np.isnan(seen[self.features.kind.eq('late')]).all())
        self.assertTrue(np.isfinite(seen[self.features.kind.eq('ordinary')]).all())

    def test_observed_counts_are_preserved(self):
        order = arrival_order(self.features, 'random', 123)
        seen = reveal(self.prelim, order, 0.25)
        known = np.isfinite(seen).all(axis=1)
        predicted = simple_forecast(self.features, seen, 'county_swing')
        np.testing.assert_array_equal(predicted[known], seen[known])
        self.assertTrue(np.isfinite(predicted).all())
        self.assertTrue((predicted >= 0).all())

    def test_regression_and_startup_blend_have_no_future_leakage(self):
        order = arrival_order(self.features, 'random', 5)
        seen = reveal(self.prelim, order, 0.01)
        changed_future = self.prelim.copy()
        changed_future[~np.isfinite(seen).all(axis=1)] = 999999
        model = SwingRegression(self.features, transform='share', early_guard=True)
        prediction = model.fit_predict(seen)
        repeated = model.fit_predict(reveal(changed_future, order, 0.01))
        np.testing.assert_array_equal(prediction, repeated)
        known = np.isfinite(seen).all(axis=1)
        np.testing.assert_array_equal(prediction[known], seen[known])
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction >= 0).all())


if __name__ == '__main__':
    unittest.main()
