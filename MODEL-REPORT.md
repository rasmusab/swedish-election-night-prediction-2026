# Election-night forecast: implementation and backtest

The selected model is a fast, interpretable regression of party-share changes,
with geographical shrinkage and a separate vote-volume regression. It improves
on simple swing benchmarks after the earliest returns in this 2018→2022 backtest.
The evidence supports using it as a working forecasting baseline. It does not
establish calibrated probabilities for the 2026 election.

![Backtest performance](outputs/backtest-performance.png)

## Seat projection added 7 September 2026

The unchanged vote model now feeds a constituency-based Riksdag allocator.
Official 2026 geography and fixed seats are frozen with the baseline. The HTML
report leads with party seats, fixed/adjustment-seat detail, and V+S+MP+C versus
M+L+KD+SD, marking the 175-seat majority threshold.

The allocator exactly reproduces national seats for 2018 and 2022 and every
2022 constituency’s fixed, adjustment and total seats. The frozen 2018→2022
forecast was evaluated across 312 combinations of methods, reporting fractions
and reporting-order scenarios. In the archived order, party seat MAE is 2.5 at
1% reporting, 1.0 at 5%, 0.5 at 10% and 25%, 0.25 at 50%, and zero from 75%.
The projected blocs are 174/175 at 10% versus the final 173/176. At 1% and 5%,
the projected majority is on the wrong side. Across all tested orders, the
worst bloc error at 10% is three seats; at 75% it is one seat. These scenarios
all reuse one target election and are not independent election validation.

![Seat forecast performance](outputs/seat-backtest-performance.png)

Full results, scoring definitions and limitations are in
[the seat backtest report](outputs/seat-backtest-summary.md). Run
`uv run scripts/backtest-seats.py` to reproduce it. Vote fitting plus allocation
was under 0.02 seconds per selected-model case. No model parameters changed.
Approximate Bayesian intervals have since been added; estimated majority
probabilities are displayed under the model assumptions. The displayed
seats are the allocation implied by a single vote forecast, not mean seats over
possible outcomes. ÖVR stays in threshold denominators but receives no seats;
the forecast assumes no individual party in that pool qualifies.

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

| Model | Development MAE (pp) |
| --- | --- |
| share_ridge_v1 | 0.118 |
| share_calibrated | 0.132 |
| share_more_shrinkage | 0.133 |
| share_less_shrinkage | 0.137 |
| clr_calibrated | 0.148 |
| clr_ridge_v1 | 0.186 |

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

| Ordinary districts reported | Raw reported share | County swing | Selected regression |
| --- | --- | --- | --- |
| 1% | 1.853 | 0.540 | 0.678 |
| 5% | 0.982 | 0.340 | 0.298 |
| 10% | 0.731 | 0.277 | 0.198 |
| 25% | 0.398 | 0.156 | 0.094 |
| 50% | 0.117 | 0.056 | 0.036 |
| 90% | 0.092 | 0.022 | 0.019 |
| 100% | 0.050 | 0.024 | 0.020 |

At 10% reported, the selected model's worst party error is 0.535 pp.
The remaining individual districts are harder: their vote-weighted average party
error is 1.63 pp (1.15
for mapped districts, 2.59 for unmatched districts).
Its national valid-vote volume error at that stage is
-1.46%. Good national shares do not imply equally
accurate local predictions or turnout forecasts.

| Party | Forecast at 10% (%) | Final 2022 (%) | Error (pp) |
| --- | --- | --- | --- |
| M | 18.567 | 19.102 | -0.535 |
| C | 6.791 | 6.714 | 0.077 |
| L | 4.447 | 4.609 | -0.161 |
| KD | 5.392 | 5.337 | 0.056 |
| MP | 5.017 | 5.082 | -0.066 |
| S | 30.195 | 30.325 | -0.131 |
| V | 7.031 | 6.747 | 0.285 |
| SD | 20.978 | 20.536 | 0.442 |
| ÖVR | 1.580 | 1.548 | 0.033 |

Late votes remain unreported throughout these election-night replays. At 100%
ordinary reporting, the final-result forecast therefore still contains estimated
late votes and uncorrected preliminary returns. The archived timestamps include
later corrections (two ordinary districts dated September 22), so the replay is
not an exact first-appearance reconstruction of election night.

## Runtime

The maximum measured point fit plus prediction across 104 evaluation checkpoints
was 0.021 seconds on this machine.
The full evaluation, including four benchmarks, took 2.00 seconds
after input loading/design construction. The tested 2026 rehearsal integration
also completed in under one second including loading and output. This is far
inside the requested 15-minute fit budget. Network download time is separate.

## Current approximate Bayesian uncertainty

The live workflow now simulates joint constituency party votes, allocates seats
in every draw, and reports central 50% and 90% seat ranges. The point model is
unchanged. The covariance estimate has a prior floor for sparse early returns;
additional explicit assumptions cover missing-versus-reported differences,
late votes and final revisions. Partial and completed preliminary counts remain
fixed before the separate final-revision layer.

The current 2018→2022 evaluation contains 80 cases at 256 draws, including five
arrival orders, eight reporting stages, component removals and half/double
systematic-error scales. All default 90% party/bloc seat intervals cover their
final outcomes in these particular checks. This is conservative behaviour on one
previous election, not demonstrated 2026 calibration: archived-order 90% bloc
width is 36 seats at 10% reporting and 6 seats at 50%. No settings were selected
by maximising coverage on these outcomes. The earlier point model was selected
using 2022, so this is not an untouched election holdout.

See [the full diagnostic](outputs/predictive-uncertainty-report.md) for the
stage-specific results and sensitivity, and [UNCERTAINTY.md](UNCERTAINTY.md)
for assumptions and operational details. The report uses 4,000 draws by default
and retains the joint seat draws for inspection. Bloc majority probabilities
are the fraction of simulations with at least 175 seats. Another historical
election pair remains a substantive next check.

## Earlier uncertainty experiments

The ordinary conditional Gaussian approximation includes shared coefficient
uncertainty and district residual covariance across parties and volume. Its
nominal 90% intervals covered only 51.1% of the tested
party/checkpoint outcomes, and covered none at 100% ordinary reporting. They
omit the uncertainty in a common late-vote shift and recount revisions.

An additional sensitivity model introduces one shared national late-vote effect
with covariance equal to one reported district's residual covariance. This is an
explicit prior assumption, not a late-effect variance learned from ordinary
returns. Overall diagnostic coverage increased to 93.9%;
coverage at 10% ordinary reporting is 80.0%.
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
