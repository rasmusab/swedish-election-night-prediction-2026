# %%
"""Step 3: synthetic election-night replay with simple forecasting benchmarks.

2018 is known. Reveal 2022 preliminary district totals; score against the final
2022 result. Arrival orders are simulated, not recovered historical timestamps.
Late-vote municipality pools remain unreported at every election-night stage.
Run: uv run scripts/replay-baselines.py
"""
from val2026 import ROOT
import pandas as pd
from val2026.replay import load_backtest, arrival_order, reveal, simple_forecast, score, SCENARIOS, FRACTIONS

# %%
features, preliminary, final = load_backtest()
rows = []
for scenario in SCENARIOS:
    for seed in ([0] if scenario == 'archived_report_time' else [0, 1, 2]):
        order = arrival_order(features, scenario, seed)
        for fraction in FRACTIONS:
            observed = reveal(preliminary, order, fraction)
            for method in ['raw_reported', 'historical', 'national_swing', 'county_swing']:
                predicted = simple_forecast(features, observed, method)
                metrics, _ = score(features, observed, predicted, final)
                if method == 'raw_reported':
                    # A share-only benchmark has no meaningful turnout prediction.
                    metrics['valid_vote_volume_error_pct'] = float('nan')
                rows.append(dict(scenario=scenario, seed=seed, fraction=fraction, model=method, **metrics))
results = pd.DataFrame(rows)
output = ROOT / 'outputs'
output.mkdir(exist_ok=True)
results.to_csv(output / 'baseline-replay.csv', index=False)
print('National party MAE (percentage points), averaged over synthetic orders:')
print(results.groupby(['fraction', 'model']).mae_pp.mean().unstack().round(3).to_string())
print('\nLate votes remain forecast at 100% ordinary-district reporting.')
