"""Check scoring independently of the predictive simulator being evaluated."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('predictive_backtest', ROOT / 'scripts/backtest-uncertainty.py')
backtest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backtest)


class PredictiveScoringTests(unittest.TestCase):
    def test_discrete_intervals_preserve_threshold_gap_and_integer_endpoints(self):
        samples = np.array([[0]] * 30 + [[15]] * 70)
        rows = backtest.interval_metrics(samples, [15], [15], ['L'], 'party_seats', discrete=True)
        wide = next(row for row in rows if row['nominal'] == .9)
        self.assertEqual((wide['lower'], wide['upper']), (0, 15))
        self.assertTrue(wide['covered'])
        self.assertEqual(wide['mean'], 10.5)
        self.assertEqual(wide['median'], 15)

    def test_majority_is_calculated_inside_each_joint_draw(self):
        parties = backtest.PARTIES
        seats = np.zeros((4, len(parties)), dtype=int)
        seats[:, parties.index('S')] = [174, 175, 173, 176]
        seats[:, parties.index('M')] = 349 - seats[:, parties.index('S')]
        actual = dict.fromkeys(parties, 0)
        actual.update(S=173, M=176)
        shares = seats / 349
        final_shares = np.array([actual[p] for p in parties]) / 349
        rows = backtest.event_metrics(seats, actual, shares, final_shares)
        left = next(row for row in rows if row['event'] == 'majority' and row['entity'] == 'V+S+MP+C')
        right = next(row for row in rows if row['event'] == 'majority' and row['entity'] == 'M+L+KD+SD')
        self.assertEqual(left['probability'], .5)
        self.assertEqual(right['probability'], .5)
        self.assertEqual(left['brier'], .25)
        self.assertFalse(left['actual'])
        self.assertTrue(right['actual'])

    def test_interval_miss_records_distance_in_metric_units(self):
        rows = backtest.interval_metrics(np.full((20, 1), 170), [173], [170], ['bloc'],
                                        'bloc_seats', discrete=True)
        self.assertTrue(all(row['width'] == 0 and row['distance_outside'] == 3
                            and not row['covered'] for row in rows))


if __name__ == '__main__':
    unittest.main()
