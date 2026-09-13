"""Regression fixtures derived from archived official results, not the allocator."""
import json
from pathlib import Path
import unittest

from val2026.seats import allocate_riksdag

ROOT = Path(__file__).resolve().parents[1]


class HistoricalSeatTests(unittest.TestCase):
    def test_official_2018_and_2022_national_and_2022_constituency_seats(self):
        for year in (2018, 2022):
            with self.subTest(year=year):
                fixture = json.loads((ROOT / f'data/seats/history/{year}.json').read_text())
                allocation = allocate_riksdag(fixture['votes'], fixture['fixed_seats'])
                for party, count in fixture['national_seats'].items():
                    self.assertEqual(allocation['national_seats'].get(party, 0), count)
                for code, expected in fixture.get('constituencies', {}).items():
                    for kind in ('fixed', 'adjustment', 'total'):
                        for party, count in expected[kind].items():
                            self.assertEqual(allocation['constituencies'][code][kind].get(party, 0), count)


if __name__ == '__main__':
    unittest.main()
