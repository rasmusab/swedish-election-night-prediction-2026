# %%
"""Inspect the latest verified snapshot without downloading; handles waiting states.

uv run scripts/forecast-latest.py --environment production
For collection and an HTML report use update-report.py instead.
"""
import argparse
import pandas as pd
# Preserve the reusable parsing interface for earlier notebooks and tests.
from val2026.election_forecast import (PARTY_CODES, reporting_counts, align_observed,
                               preserve_partial_lower_bounds, load_snapshot)
from val2026.election_report import ROOT, run_cycle, update_lock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=['production','rehearsal'], default='rehearsal')
    args = parser.parse_args(argv)
    with update_lock(ROOT):
        state = run_cycle(ROOT, args.environment, collect=False)
    print('TEST DATA — artificial rehearsal results' if state['test'] else 'Production preliminary count')
    print(state['health'], state['message'])
    if state['parties']:
        print(pd.DataFrame(state['parties']).round(4).to_string(index=False))
    return state


# %%
if __name__ == '__main__':
    prediction = main()
