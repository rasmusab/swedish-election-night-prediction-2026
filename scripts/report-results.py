# %%
"""Render the saved backtest results and write the findings. No models refitted."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/val2026-matplotlib')
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from val2026 import ROOT
OUT = ROOT / 'outputs'
metrics = pd.read_csv(OUT / 'model-evaluation-metrics.csv')
parties = pd.read_csv(OUT / 'model-evaluation-parties.csv')
config = json.loads((OUT / 'selected-model.json').read_text())
selected = config['selected_name']
labels = {'raw_reported': 'Raw reported share', 'county_swing': 'County swing', selected: 'Selected regression'}
colors = {'raw_reported': '#9199a6', 'county_swing': '#cf7d24', selected: '#17679a'}
archived = metrics[metrics.scenario.eq('archived_report_time')]

# %% Static exportable comparison figure.
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'axes.labelcolor': '#334155', 'text.color': '#172335'})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
for model, label in labels.items():
    d = archived[archived.model.eq(model)]
    axes[0].plot(d.fraction, d.mae_pp, marker='o', ms=4, label=label, color=colors[model], lw=2)
axes[0].set(xlabel='Ordinary districts reported', ylabel='Mean absolute party error (percentage points)',
            title='Archived 2022 reporting order')
axes[0].xaxis.set_major_formatter(PercentFormatter(1))
axes[0].grid(axis='y', alpha=.18)
axes[0].legend(frameon=False)
ten = metrics[metrics.fraction.eq(.1)].groupby(['scenario', 'model']).mae_pp.mean().unstack()
order = ['archived_report_time', 'random', 'small_first', 'geography_delayed', 'big_cities_late']
nice = ['Archived times', 'Random', 'Small districts first', 'Geographic delays', 'Large cities late']
y = np.arange(len(order))
for i, (model, label) in enumerate(labels.items()):
    axes[1].barh(y + (i - 1) * .23, ten.loc[order, model], height=.22, color=colors[model], label=label)
axes[1].set(yticks=y, yticklabels=nice, xlabel='Mean absolute party error (percentage points)', title='At 10% of ordinary districts reported')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=.18)
fig.suptitle('2018 history → 2022 incoming returns', fontsize=17, fontweight='bold')
fig.savefig(OUT / 'backtest-performance.png', dpi=180)
fig.savefig(OUT / 'backtest-performance.svg')
plt.close(fig)

# %% Markdown table utility without an additional package dependency.
def table(frame):
    rows = [[str(c) for c in frame.columns]] + frame.astype(str).values.tolist()
    return '\n'.join(['| ' + ' | '.join(rows[0]) + ' |', '| ' + ' | '.join(['---'] * len(rows[0])) + ' |'] +
                     ['| ' + ' | '.join(row) + ' |' for row in rows[1:]])

comparison = archived[archived.model.isin(labels)].pivot(index='fraction', columns='model', values='mae_pp').loc[[.01,.05,.1,.25,.5,.9,1], list(labels)]
comparison.columns = [labels[c] for c in comparison.columns]
comparison = comparison.map(lambda x: f'{x:.3f}')
comparison.insert(0, 'Ordinary districts reported', [f'{i:.0%}' for i in comparison.index])
ten_parties = parties[parties.scenario.eq('archived_report_time') & parties.model.eq(selected) & parties.fraction.eq(.1)]
party_table = ten_parties[['party', 'predicted_share_pct', 'final_share_pct', 'error_pp']].copy()
for c in party_table.columns[1:]:
    party_table[c] = party_table[c].map(lambda x: f'{x:.3f}')
party_table.columns = ['Party', 'Forecast at 10% (%)', 'Final 2022 (%)', 'Error (pp)']
selected_metrics = metrics[metrics.model.eq(selected)]
ordinary_interval = json.loads((OUT / 'uncertainty-summary.json').read_text())
late_interval = json.loads((OUT / 'uncertainty-with-late-shift-summary.json').read_text())
runtime = json.loads((OUT / 'model-evaluation-config.json').read_text())['total_runtime_seconds']
arch10 = archived[archived.model.eq(selected) & archived.fraction.eq(.1)].iloc[0]
dev = pd.read_csv(OUT / 'model-iteration-metrics.csv')
ranking = dev[dev.model.isin(config['development_scores_pp']) & dev.fraction.isin([.05,.1,.25,.5])].groupby('model').mae_pp.mean().sort_values()
ranking_table = pd.DataFrame({'Model': ranking.index, 'Development MAE (pp)': [f'{v:.3f}' for v in ranking]})
report = f'''# Election-night forecast: implementation and backtest

The selected model is a fast, interpretable regression of party-share changes,
with geographical shrinkage and a separate vote-volume regression. It improves
on simple swing benchmarks after the earliest returns in this 2018→2022 backtest.
The evidence supports using it as a working forecasting baseline. It does not
establish calibrated probabilities for the 2026 election.

![Backtest performance](outputs/backtest-performance.png)

## Seat projections

The unchanged vote forecast also feeds a constituency-based Riksdag allocator,
using frozen official 2026 geography and fixed seats. The live report leads with
party seats and V+S+MP+C versus M+L+KD+SD, with 175 seats marking a majority.
The independently implemented allocator handles thresholds, returned fixed
seats and adjustment seats. ÖVR contributes to threshold denominators but is
excluded from seats; no individual party within that pool is assumed to qualify.
Approximate Bayesian joint simulations now supply central 50%/90% seat ranges.
The point projection is unchanged; estimated bloc majority probabilities are displayed
under the model assumptions. See [UNCERTAINTY.md](UNCERTAINTY.md).

Run `uv run scripts/backtest-seats.py` for historical allocation validation and
rolling seat scores. Full results are in
[the seat backtest report](outputs/seat-backtest-summary.md).
Run `uv run scripts/backtest-uncertainty.py --draws 256` for the current
[joint vote/seat uncertainty diagnostic](outputs/predictive-uncertainty-report.md),
including reporting-order stress tests and systematic-scale sensitivity.

![Seat forecast performance](outputs/seat-backtest-performance.png)

## Work completed in order

1. Downloaded and reconciled actual final and preliminary 2022 results. Prepared
   the 2026 historical baseline from real 2022 votes and pre-election 2026
   electorate counts. No rehearsal votes enter that baseline.
2. Handled changed districts using official comparability flags, summed old
   districts, residual historical municipality pools, and explicit fallbacks.
   All 1,253 unmatched 2026 districts receive a baseline. Collection districts
   form 290 separate municipality pools.
3. Built replay scenarios that reveal preliminary counts while withholding final
   outcomes from every fit. Added archived reporting timestamps and stress orders.
4. Implemented a bounded collector with separate rehearsal/production and
   preliminary/final storage, immutable snapshots, change detection, official
   signature verification, retry/backoff, and stale-data flags. Rehearsal succeeds;
   the production index returned HTTP 404 when checked, and is handled cleanly.

See PREPARATION.md and data/SOURCES.md for source and mapping details. Two source
issues are corrected: the preliminary Excel subtotal includes 3,481 invalid
unregistered-party votes, and its district results for 01270007/01270008 are
swapped. Replay uses the corrected official preliminary JSON. The final workbook
and final archive agree at every district/party, with 6,477,970 valid votes.

## Model

For district i and party p, regress the change in vote share from the historical
baseline on historical party shares, log electorate, historical valid-vote rate,
an unmatched-district indicator, county effects and municipality effects.
The regression has Gaussian shrinkage priors: its conditional MAP is the ridge
solution `(X'WX + Lambda)^-1 X'WY`. Default penalties are 20 for standardized
features, 10 for county effects, 10 for municipality effects, and 0.001 for the
national intercept. Weights scale with the square root of reported valid votes
and are capped to limit dominance by large districts.

Raw share predictions are clipped to a small positive value and renormalized.
The volume model regresses log(reported valid votes / historical expected votes)
on the same covariates; its predicted ordinary-district volume cannot exceed
known electorate. The historical expectation uses current electorate and prior
valid-vote rates, never the missing district's actual current-election count.
Reported counts are retained exactly. Late pools use their own historical offsets
and ordinary-municipality covariates for the estimated swing.

The first 100 ordinary reports use an equal-weight average of regression and
county swing. The regression weight then rises linearly to 100% by 300 reports.
This startup safeguard was added after the archived order exposed an early
regression weakness. It reduces extrapolation from the nonrandom first reports.

This is Gaussian-prior regression with a direct numerical solution, not black-box
machine learning. Full Stan MCMC is unnecessary for this first implementation:
the point estimate is already available analytically. Stan would be useful for
learning hierarchical variances, robust residual distributions and a richer
late-vote model. See the primary [Stan regression guide](https://mc-stan.org/docs/stan-users-guide/regression.html).

## Iteration and evaluation

Started with raw-share and centered-log-ratio regressions. Then tested calibration
on the revealed aggregate and weaker/stronger shrinkage. Raw-share regression
won the development objective (mean party MAE at 5%, 10%, 25%, 50% reporting):

{table(ranking_table)}

Calibration improved the log-ratio model but did not improve raw-share regression,
so it was not retained. The initial model was selected on synthetic development
seeds 0–2, then tested on seeds 100–102 and the archived order. The startup revision
was checked on seeds 200–202. A complete fallback to county swing was less useful
than an equal-weight startup blend; that comparison is retained in outputs.

These are different reporting orders of the SAME 2022 election, not independent
future-election validation. The archived order informed the startup revision and
therefore is no longer an untouched holdout. No within-checkpoint fit sees future
returns, but retrospective model selection can still overfit this one election.

## Accuracy

Mean absolute error across the eight parliamentary parties plus ÖVR, in percentage
points of national valid votes, using the archived reporting order:

{table(comparison.reset_index(drop=True))}

At 10% reported, the selected model's worst party error is {arch10.max_error_pp:.3f} pp.
The remaining individual districts are harder: their vote-weighted average party
error is {arch10.remaining_district_mae_pp:.2f} pp ({arch10.comparable_district_mae_pp:.2f}
for mapped districts, {arch10.unmatched_district_mae_pp:.2f} for unmatched districts).
Its national valid-vote volume error at that stage is
{arch10.valid_vote_volume_error_pct:.2f}%. Good national shares do not imply equally
accurate local predictions or turnout forecasts.

{table(party_table)}

Late votes remain unreported throughout these election-night replays. At 100%
ordinary reporting, the final-result forecast therefore still contains estimated
late votes and uncorrected preliminary returns. The archived timestamps include
later corrections (two ordinary districts dated September 22), so the replay is
not an exact first-appearance reconstruction of election night.

## Runtime

The maximum measured point fit plus prediction across 104 evaluation checkpoints
was {selected_metrics.fit_predict_seconds.max():.3f} seconds on this machine.
The full evaluation, including four benchmarks, took {runtime:.2f} seconds
after input loading/design construction. The tested 2026 rehearsal integration
also completed in under one second including loading and output. This is far
inside the requested 15-minute fit budget. Network download time is separate.

## Earlier uncertainty experiments

The ordinary conditional Gaussian approximation includes shared coefficient
uncertainty and district residual covariance across parties and volume. Its
nominal 90% intervals covered only {ordinary_interval['coverage']:.1%} of the tested
party/checkpoint outcomes, and covered none at 100% ordinary reporting. They
omit the uncertainty in a common late-vote shift and recount revisions.

An additional sensitivity model introduces one shared national late-vote effect
with covariance equal to one reported district's residual covariance. This is an
explicit prior assumption, not a late-effect variance learned from ordinary
returns. Overall diagnostic coverage increased to {late_interval['coverage']:.1%};
coverage at 10% ordinary reporting is {late_interval['coverage_by_fraction']['0.1']:.1%}.
In the large-cities-late scenario, early 10% coverage is only 44%, so the model
still misses systematic reporting-order risk. Late-stage intervals can instead
be conservative. Recount revisions are still not explicitly modeled.

Repeated stages, scenarios and party outcomes are dependent. These coverage
percentages are diagnostic, not evidence that a stated 90% probability will
calibrate in 2026. Do not derive confident winner probabilities from them.

## Re-run and use

Run `uv run scripts/backtest-model.py` for the frozen-model evaluation. Use
`uv run scripts/check-uncertainty.py --late-shift` for the sensitivity intervals and
`uv run scripts/report-results.py` to refresh this report/figure. The scripts use `# %%`
cells and may also be run from an interactive Python editor.

`uv run scripts/collect-results.py --environment production --once` checks production;
`uv run scripts/forecast-latest.py --environment production` forecasts the last verified
preliminary snapshot. The rehearsal equivalents run without the environment
flag and always label test data. No background monitor is running.

The integration preserves all complete observed district/pool counts. Partial
late pools are kept distinct from complete ones, with known counts recorded and
used as a lower bound. Their within-pool allocation is not a separate fitted
model. Ordinary geography and constituency membership come from frozen
official 2026 inputs. A complete signed feed establishes the late-record roster
separately for each environment. Unexpected code/type/constituency changes fail
clearly and retain the previous forecast pending review.

The next substantive validation is another election pair, plus a model for
preliminary-to-final revisions and late-vote shifts. Those are the main limits
on turning this working prototype into a probability forecast.
'''
(ROOT / 'MODEL-REPORT.md').write_text(report)
print('Wrote MODEL-REPORT.md and outputs/backtest-performance.png/.svg')
