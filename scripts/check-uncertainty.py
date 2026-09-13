# %%
"""Evaluate conditional intervals against the retrospective 2022 target.

Run: uv run scripts/check-uncertainty.py --draws 200
These are conditional model intervals, excluding recount revisions and an
extra national late-vote shift. They are not calibrated winning probabilities.
2022 final outcomes enter scoring only, never fitting or interval construction.
"""
import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from val2026.election_data import PARTIES
from val2026.forecast_model import SwingRegression
from val2026.replay import load_backtest, arrival_order, reveal, SCENARIOS
from scripts.uncertainty import conditional_share_draws, interval_scope

from val2026 import ROOT
OUT = ROOT / 'outputs'


# %% Fit intervals without passing final or unrevealed outcomes to the model.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draws', type=int, default=200)
    parser.add_argument('--late-shift', action='store_true', help='Add an explicitly assumed common late-vote effect')
    args = parser.parse_args(argv)
    config = json.loads((OUT / 'selected-model.json').read_text())
    model_name, options = next(iter(config['models'].items()))
    features, preliminary, final = load_backtest()
    model = SwingRegression(features, **options)
    final_shares = final.sum(axis=0) / final.sum()
    rows, diagnostic_rows = [], []
    for scenario_index, scenario in enumerate(SCENARIOS):
        order = arrival_order(features, scenario, 100)
        for stage_index, fraction in enumerate([0.10, 0.50, 0.90, 1.00]):
            observed = reveal(preliminary, order, fraction)
            result = conditional_share_draws(model, observed, draws=args.draws,
                seed=2026 + scenario_index * 100 + stage_index, late_shift=args.late_shift)
            keys = {'model': model_name, 'scenario': scenario, 'seed': 100,
                    'fraction': fraction, 'nominal_coverage': 0.90}
            for party, estimate, lower, upper, target in zip(
                    PARTIES, result.map_shares, result.lower, result.upper, final_shares):
                rows.append({**keys, 'party': party, 'map_share_pct': estimate * 100,
                    'lower_share_pct': lower * 100, 'upper_share_pct': upper * 100,
                    'final_share_pct': target * 100, 'width_pp': (upper - lower) * 100,
                    'covered': bool(lower <= target <= upper)})
            diagnostic_rows.append({**keys, **result.diagnostics})
            print(f'{scenario} {fraction:.0%}: '
                  f'{np.mean((result.lower <= final_shares) & (final_shares <= result.upper)):.0%} '
                  f'party coverage; {result.diagnostics["seconds"]:.2f}s', flush=True)
    results = pd.DataFrame(rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    OUT.mkdir(exist_ok=True)
    prefix = 'uncertainty-with-late-shift' if args.late_shift else 'uncertainty'
    results.to_csv(OUT / f'{prefix}-coverage.csv', index=False)
    diagnostics.to_csv(OUT / f'{prefix}-diagnostics.csv', index=False)
    by_fraction = results.groupby('fraction').agg(coverage=('covered', 'mean'), width_pp=('width_pp', 'mean'))
    by_scenario = results.groupby('scenario').agg(coverage=('covered', 'mean'), width_pp=('width_pp', 'mean'))
    summary = {
        'scope': interval_scope(args.late_shift), 'model': model_name, 'draws_per_checkpoint': args.draws,
        'shared_late_shift_prior': args.late_shift,
        'late_shift_assumption': 'When enabled, add one national late share/volume shift with the reported-district residual covariance. Its variance is assumed, not learned from ordinary returns.',
        'coverage': float(results.covered.mean()),
        'mean_width_pp': float(results.width_pp.mean()),
        'max_checkpoint_seconds': float(diagnostics.seconds.max()),
        'coverage_by_fraction': {str(k): float(v) for k, v in by_fraction.coverage.items()},
        'coverage_by_scenario': {str(k): float(v) for k, v in by_scenario.coverage.items()},
        'evaluation_limit': 'One retrospective election; reporting orders and party outcomes are dependent. '
                            'Coverage is diagnostic and does not establish future-election calibration.',
    }
    (OUT / f'{prefix}-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('\n' + interval_scope(args.late_shift))
    print('\nCoverage and average 90% interval width (pp), by reporting fraction:')
    print(by_fraction.round(3).to_string())
    print('\nBy arrival scenario:')
    print(by_scenario.round(3).to_string())
    print(f'\nMaximum checkpoint runtime: {diagnostics.seconds.max():.2f}s')
    return results, diagnostics


# %%
if __name__ == '__main__':
    coverage, diagnostics = main()
