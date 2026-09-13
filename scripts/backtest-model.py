# %%
"""Test fast Gaussian-prior swing regressions on the 2018→2022 election.

Run: uv run scripts/backtest-model.py
For the initial comparison: uv run scripts/backtest-model.py --phase development
All fit inputs use only 2018 history, known 2022 electorate, and revealed
2022 preliminary returns. Final 2022 results are passed only to scoring.
Run cell-by-cell with main(['--phase', 'development']).
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/val2026-matplotlib')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from val2026.election_data import PARTIES
from val2026.replay import load_backtest, arrival_order, reveal, simple_forecast, score, FRACTIONS, SCENARIOS
from val2026.forecast_model import SwingRegression

from val2026 import ROOT
OUT = ROOT / 'outputs'
INITIAL_MODELS = {
    'clr_ridge_v1': dict(transform='clr'),
    'share_ridge_v1': dict(transform='share'),
}
ITERATION_MODELS = {
    **INITIAL_MODELS,
    'clr_calibrated': dict(transform='clr', calibrate=True),
    'share_calibrated': dict(transform='share', calibrate=True),
    'share_less_shrinkage': dict(transform='share', calibrate=True, feature_penalty=5., county_penalty=3., municipality_penalty=3.),
    'share_more_shrinkage': dict(transform='share', calibrate=True, feature_penalty=80., county_penalty=30., municipality_penalty=30.),
}


# %% Fit every checkpoint afresh and record both accuracy and measured runtime.
def run(models, phase, seeds, scenarios):
    features, preliminary, final = load_backtest()
    regressions = {name: SwingRegression(features, **options) for name, options in models.items()}
    metrics_rows, party_rows = [], []
    started = perf_counter()
    for scenario in scenarios:
        scenario_seeds = [0] if scenario == 'archived_report_time' else seeds
        for seed in scenario_seeds:
            order = arrival_order(features, scenario, seed)
            for fraction in FRACTIONS:
                observed = reveal(preliminary, order, fraction)
                for name in ['raw_reported', 'historical', 'national_swing', 'county_swing', *models]:
                    t0 = perf_counter()
                    if name in regressions:
                        predicted, details = regressions[name].fit_predict(observed, return_details=True)
                        seconds = details['seconds']
                    else:
                        predicted = simple_forecast(features, observed, name)
                        seconds = perf_counter() - t0
                    metrics, error = score(features, observed, predicted, final)
                    if name == 'raw_reported':
                        metrics['valid_vote_volume_error_pct'] = np.nan
                    keys = dict(scenario=scenario, seed=seed, fraction=fraction, model=name)
                    metrics_rows.append({**keys, **metrics, 'fit_predict_seconds': seconds})
                    pred_share = predicted.sum(axis=0) / predicted.sum()
                    truth_share = final.sum(axis=0) / final.sum()
                    for p, estimate, truth, err in zip(PARTIES, pred_share, truth_share, error):
                        party_rows.append({**keys, 'party': p, 'predicted_share_pct': estimate * 100,
                                           'final_share_pct': truth * 100, 'error_pp': err})
            print(f'{phase}: {scenario}, seed {seed} complete', flush=True)
    metrics = pd.DataFrame(metrics_rows)
    OUT.mkdir(exist_ok=True)
    metrics.to_csv(OUT / f'model-{phase}-metrics.csv', index=False)
    pd.DataFrame(party_rows).to_csv(OUT / f'model-{phase}-parties.csv', index=False)
    (OUT / f'model-{phase}-config.json').write_text(json.dumps({'models': models, 'seeds': seeds,
        'scenarios': scenarios, 'total_runtime_seconds': perf_counter() - started}, indent=2) + '\n')
    print('\nNational party MAE (pp):')
    print(metrics.groupby(['fraction', 'model']).mae_pp.mean().unstack().round(3).to_string())
    print('\nMax fit + prediction time (seconds):')
    print(metrics.groupby('model').fit_predict_seconds.max().round(3).to_string())
    return metrics


# %%
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['development', 'iteration', 'evaluation', 'startup-check'], default='evaluation')
    args = parser.parse_args(argv)
    if args.phase == 'development':
        # Initial comparison excludes the archived order and reserved seeds.
        return run(INITIAL_MODELS, args.phase, [0, 1, 2], [s for s in SCENARIOS if s != 'archived_report_time'])
    if args.phase == 'iteration':
        metrics = run(ITERATION_MODELS, args.phase, [0, 1, 2], [s for s in SCENARIOS if s != 'archived_report_time'])
        # Select using only synthetic development replays. Freeze before running
        # archived order or reserved reporting-order seeds. These remain the
        # same election, so they are not an independent election-level holdout.
        ranking = metrics[metrics.model.isin(ITERATION_MODELS) & metrics.fraction.isin([0.05, 0.10, 0.25, 0.50])].groupby('model').mae_pp.mean().sort_values()
        winner = ranking.index[0]
        selection = {'selected_name': winner, 'models': {winner: ITERATION_MODELS[winner]},
                     'selection': 'Mean party MAE over 5, 10, 25, 50% on synthetic development seeds 0–2',
                     'development_scores_pp': ranking.to_dict(),
                     'limitation': 'Reserved reporting orders use the same 2022 election, not independent election outcomes.'}
        (OUT / 'selected-model.json').write_text(json.dumps(selection, indent=2) + '\n')
        print('\nFrozen selection:', winner)
        return metrics
    config = json.loads((OUT / 'selected-model.json').read_text())
    if args.phase == 'startup-check':
        metrics = run({'share_ridge_v1': dict(transform='share'),
                    'guarded_share_ridge': dict(transform='share', early_guard=True)},
                   args.phase, [200, 201, 202], SCENARIOS)
        config['selected_name'] = 'guarded_share_ridge'
        config['models'] = {'guarded_share_ridge': dict(transform='share', early_guard=True)}
        config['startup_revision'] = 'After archived-order weakness at 1%, equal-weight county swing/regression through 100 reports, full regression by 300. Retrospective sensitivity check on same election.'
        (OUT / 'selected-model.json').write_text(json.dumps(config, indent=2) + '\n')
        return metrics
    return run(config['models'], args.phase, [100, 101, 102], SCENARIOS)


if __name__ == '__main__':
    metrics = main()
