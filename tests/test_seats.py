import json
import math
from pathlib import Path
import random
import unittest

from val2026.seats import AllocationError, allocate_riksdag


class RiksdagSeatTests(unittest.TestCase):
    def assert_reconciles(self, result, fixed):
        self.assertEqual(sum(result['national_seats'].values()), result['total_seats'])
        self.assertEqual(sum(result['national_fixed_seats'].values()), sum(fixed.values()))
        self.assertEqual(sum(result['national_adjustment_seats'].values()),
                         result['total_seats'] - sum(fixed.values()))
        for code, row in result['constituencies'].items():
            self.assertEqual(sum(row['fixed'].values()), fixed[code])
            for party in result['national_seats']:
                self.assertEqual(row['total'][party], row['fixed'][party] + row['adjustment'][party])
                self.assertGreaterEqual(row['fixed'][party], 0)
                self.assertGreaterEqual(row['adjustment'][party], 0)
        json.dumps(result, allow_nan=False)

    def test_complete_349_seats_and_fractional_weights(self):
        result = allocate_riksdag({'01': {'A': 60.125, 'B': 39.875}}, {'01': 310})
        self.assertEqual(result['national_seats'], {'A': 210, 'B': 139})
        self.assertEqual(result['national_vote_totals']['A'], 60.125)
        self.assert_reconciles(result, {'01': 310})

    def test_four_percent_threshold_is_inclusive_without_rounding(self):
        exact = allocate_riksdag({'01': {'A': 96, 'X': 4}}, {'01': 310})
        below = allocate_riksdag({'01': {'A': 96.000000001, 'X': 3.999999999}}, {'01': 310})
        self.assertIn('X', exact['national_eligible'])
        self.assertEqual(exact['national_seats']['X'], 14)
        self.assertNotIn('X', below['national_eligible'])
        self.assertEqual(below['national_seats']['X'], 0)

    def test_other_votes_count_in_threshold_but_never_receive_seats(self):
        result = allocate_riksdag({'01': {'A': 76.5, 'X': 3.5, 'ÖVR': 20}}, {'01': 310})
        self.assertEqual(result['national_seats'], {'A': 349, 'X': 0, 'ÖVR': 0})
        self.assertEqual(result['national_vote_shares']['X'], .035)
        self.assertNotIn('ÖVR', result['national_eligible'])
        self.assertNotIn('ÖVR', result['local_only_eligible'])

    def test_twelve_percent_only_allows_fixed_seats_in_qualifying_constituency(self):
        votes = {'small': {'A': 43, 'B': 45, 'R': 12},
                 'large': {'A': 450, 'B': 450, 'R': 0}}
        fixed = {'small': 10, 'large': 17}
        result = allocate_riksdag(votes, fixed, total_seats=30)
        self.assertNotIn('R', result['national_eligible'])
        self.assertEqual(result['local_only_eligible']['R'], ['small'])
        self.assertEqual(result['national_seats']['R'], 1)
        self.assertEqual(result['national_adjustment_seats']['R'], 0)
        self.assertEqual(result['constituencies']['large']['total']['R'], 0)
        self.assertEqual(sum(result['national_seats'][p] for p in ['A', 'B']), 29)
        self.assert_reconciles(result, fixed)
        votes['small'] = {'A': 43.000000001, 'B': 45, 'R': 11.999999999}
        below = allocate_riksdag(votes, fixed, total_seats=30)
        self.assertEqual(below['national_seats']['R'], 0)
        self.assertNotIn('R', below['local_only_eligible'])

    def test_returns_remove_lowest_winning_quotient_then_redistribute(self):
        votes = {'x': {'A': 100, 'B': 10}, 'y': {'A': 0, 'B': 1000}}
        fixed = {'x': 5, 'y': 5}
        result = allocate_riksdag(votes, fixed, total_seats=11)
        self.assertEqual(result['initial_fixed_seats']['x'], {'A': 5, 'B': 0})
        self.assertEqual(result['national_seats'], {'A': 1, 'B': 10})
        self.assertEqual(result['constituencies']['x']['fixed'], {'A': 1, 'B': 4})
        self.assertEqual([event['winning_quotient'] for event in result['returned_seats']],
                         [100 / 9, 100 / 7, 100 / 5, 100 / 3])
        self.assertEqual([event['to_party'] for event in result['redistributed_seats']], ['B'] * 4)
        self.assert_reconciles(result, fixed)

    def test_returned_seats_are_redistributed_globally_not_in_constituency_order(self):
        votes = {'x': {'A': 100, 'B': 20, 'C': 5},
                 'y': {'A': 200, 'B': 40, 'C': 100},
                 'z': {'A': 0, 'B': 2000, 'C': 2500}}
        fixed = {'x': 5, 'y': 5, 'z': 10}
        result = allocate_riksdag(votes, fixed, total_seats=22)
        self.assertEqual(result['national_seats'], {'A': 1, 'B': 9, 'C': 12})
        redistributed = result['redistributed_seats']
        self.assertEqual([(e['constituency'], e['to_party']) for e in redistributed[:2]],
                         [('y', 'B'), ('y', 'C')])
        # B reaches its national nine-seat cap. C then receives both adjustment
        # seats even though B has a larger next quotient in the large district.
        self.assertEqual(result['national_adjustment_seats'], {'A': 0, 'B': 0, 'C': 2})
        self.assert_reconciles(result, fixed)

    def test_fewer_than_three_fixed_seats_are_protected_from_returns(self):
        votes = {'small': {'A': 100, 'B': 0}, 'large': {'A': 0, 'B': 1000}}
        fixed = {'small': 2, 'large': 8}
        result = allocate_riksdag(votes, fixed, total_seats=11)
        self.assertEqual(result['initial_national_entitlement'], {'A': 1, 'B': 10})
        self.assertEqual(result['protected_overhang'], {'A': 1})
        self.assertEqual(result['national_seats'], {'A': 2, 'B': 9})
        self.assertEqual(result['returned_seats'], [])
        self.assert_reconciles(result, fixed)

    def test_first_adjustment_in_constituency_uses_divisor_one(self):
        votes = {'x': {'A': 300, 'B': 1000}, 'y': {'A': 110, 'B': 1000}}
        result = allocate_riksdag(votes, {'x': 3, 'y': 3}, total_seats=12)
        self.assertEqual(result['national_seats'], {'A': 2, 'B': 10})
        self.assertEqual(result['national_fixed_seats']['A'], 1)
        # x's next quotient is 300/3=100; y wins with 110/1=110.
        # Incorrectly using 1.2 gives 91.67 and would put the seat in x.
        self.assertEqual(result['constituencies']['y']['adjustment']['A'], 1)
        self.assertEqual(result['constituencies']['x']['adjustment']['A'], 0)

    def test_exact_ties_are_recorded_reproducible_and_input_order_independent(self):
        one = allocate_riksdag({'x': {'A': 50, 'B': 50}}, {'x': 1}, total_seats=1, tie_seed=7)
        two = allocate_riksdag({'x': {'B': 50, 'A': 50}}, {'x': 1}, total_seats=1, tie_seed=7)
        self.assertEqual(one, two)
        self.assertTrue(one['ties'])
        self.assertEqual(one['ties'][0]['candidates'], ['A', 'B'])
        self.assertIn(one['ties'][0]['chosen'], ['A', 'B'])
        self.assertIn('not_official', one['tie_policy'])
        winners = {tuple(allocate_riksdag({'x': {'A': 50, 'B': 50}}, {'x': 1},
                                        total_seats=1, tie_seed=seed)['national_seats'].values())
                   for seed in range(10)}
        self.assertEqual(winners, {(1, 0), (0, 1)})

    def test_near_tie_is_not_rounded_into_a_lottery(self):
        result = allocate_riksdag({'x': {'A': 50.00000000001, 'B': 50}}, {'x': 1}, total_seats=1)
        self.assertEqual(result['national_seats'], {'A': 1, 'B': 0})
        self.assertEqual(result['ties'], [])

    def test_zero_party_votes_stay_zero_but_empty_results_are_rejected(self):
        result = allocate_riksdag({'x': {'A': 100, 'Zero': 0}}, {'x': 310})
        self.assertEqual(result['national_seats'], {'A': 349, 'Zero': 0})
        for votes in [{'x': {'A': 0}}, {'x': {'A': 0, 'ÖVR': 0}}]:
            with self.assertRaises(AllocationError):
                allocate_riksdag(votes, {'x': 310})

    def test_invalid_values_missing_geography_and_no_eligible_party_fail_clearly(self):
        for bad in [math.nan, math.inf, -1, True, '10']:
            with self.subTest(bad=bad), self.assertRaises(AllocationError):
                allocate_riksdag({'x': {'A': bad}}, {'x': 310})
        for count in [-1, 1.5, True, 350]:
            with self.subTest(count=count), self.assertRaises(AllocationError):
                allocate_riksdag({'x': {'A': 10}}, {'x': count})
        with self.assertRaises(AllocationError):
            allocate_riksdag({'x': {'A': 10}}, {'y': 310})
        with self.assertRaises(AllocationError):
            allocate_riksdag({'x': {'ÖVR': 10}}, {'x': 310})

    def test_random_full_size_elections_preserve_all_margins(self):
        rng = random.Random(54)
        fixed = {str(i): (2 if i == 0 else 11) for i in range(29)}
        for case in range(5):
            votes = {c: {p: rng.uniform(1, 40) * rng.uniform(100, 1000)
                         for p in ['S', 'M', 'SD', 'V', 'C', 'L', 'MP', 'KD', 'ÖVR']}
                     for c in fixed}
            with self.subTest(case=case):
                self.assert_reconciles(allocate_riksdag(votes, fixed), fixed)


if __name__ == '__main__':
    unittest.main()
