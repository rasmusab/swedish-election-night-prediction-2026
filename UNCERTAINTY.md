# Approximate Bayesian predictive uncertainty

The live forecast retains the selected guarded share regression and its point
seat allocation. A separate simulation layer generates plausible eventual final
party votes in each constituency, allocates all 349 seats in every draw, and
summarises central 50% and 90% ranges. No Stan or new dependencies are required.

These are conditional, assumption-based predictive ranges. Fixed shrinkage and
an estimated response covariance are conditioned on; their full posterior
uncertainty is not sampled. Extra systematic-error scales are explicit priors,
not identified from the currently reported districts. The one-election checks
do not establish calibration for 2026. The report displays each bloc’s estimated probability of reaching at least
175 seats, calculated from the joint simulated seat allocations. These are
probabilities under the model assumptions, not calibrated guarantees.

## Construction and assumptions

1. **Regression coefficients.** Draw one coefficient matrix per simulated
   election, shared across all outstanding districts. Covariance is
   `(X'WX + penalty)^-1` across predictors times the joint party-share/log-volume
   residual covariance. County and municipality uncertainty is already included;
   a second set of random geographical coefficients is not added.
2. **Residual covariance.** Combine revealed ordinary-district residuals with a
   prior equivalent to 40 residual degrees of freedom. Its party-share block is
   `0.05² × (diag(p) − p p')`, where `p` is known historical national share. The
   prior log-volume standard deviation is 0.15. This prevents two early returns
   from implying almost no variance and preserves joint party/turnout variation.
3. **Outstanding ordinary variation.** Simulate joint district residuals around
   the regression. A separate common missing-versus-reported discrepancy uses
   0.5 times the residual standard deviation, multiplied by a reporting-balance
   factor between 0.25 and 1. The factor measures county, municipality-population
   and district-size imbalance using only revealed membership and known geography.
   This effect represents possible selection error outside the fitted regression;
   its scale is an explicit sensitivity assumption.
4. **Late votes.** Incomplete late pools receive local residual variation and a
   common national late effect at one residual standard deviation. Its log-volume
   standard deviation has a floor of 0.20. Counted partial votes remain a fixed
   floor in the preliminary layer. If that floor exhausts the point prediction,
   a one-sided remaining-volume allowance (up to 0.25 times historical pool
   volume as its half-normal scale) prevents an incomplete pool becoming certain.
5. **Final revisions.** After aggregating the simulated preliminary votes, add a
   separate small national revision effect. Default share scale is 0.0005
   (0.05 percentage points) and log-volume scale is 0.001 (approximately 0.1%).
   Party shifts are centred to sum to zero and the underlying normal values are
   clipped at three standard deviations before centring/normalisation. These
   limits are modelling assumptions, not certified bounds on recount changes.

Completed reported votes are exact in the **preliminary** simulation layer,
including when they exceed a frozen pre-election electorate figure. The
electorate caps only unreported ordinary estimates. The separate **final**
revision layer can alter completed preliminary votes. Thus final-result
uncertainty survives a complete preliminary count without mixing the final feed
into the application. It would need different treatment after certified results.

Party shares remain nonnegative and normalised; volumes remain nonnegative.
Clipping, volume transformation and one-sided partial-pool allowances can move
the simulation mean away from the point projection. Diagnostics record this
offset. There is no aggregate recentering that could change already reported
counts or violate a partial-pool floor. At startup, the guarded point forecast
retains full regression covariance as a conservative approximation; uncertainty
in the fallback/regression blend is not derived from a full joint posterior.

## From joint votes to seats

Each draw retains the 29 constituencies, nine party categories, turnout effects
and their dependence. The existing allocator applies the 4% national threshold,
12% local exception, returned fixed seats and adjustment seats to each draw.
The two bloc totals are calculated within that draw, never by adding party
interval endpoints. The pooled ÖVR category remains in the denominator but
receives no seats, as in the point model.

Discrete seat intervals use empirical quantiles with integer endpoints. Near
4%, a party can have separated groups of outcomes (for example zero seats or
around fifteen); the interval alone cannot describe that shape. Histograms and
draw-level seats are retained. Individual marginal medians need not sum to 349;
the headline remains the original complete point allocation.

## Validation and interpretation

`uv run scripts/backtest-uncertainty.py --draws 256` checks five reporting orders
at 1%, 5%, 10%, 25%, 50%, 75%, 90% and 100% ordinary reporting. It scores final
2022 vote shares, party seats and both blocs from the 2018 background. Final
outcomes enter scoring only. Component removals and half/double systematic-scale
checks are diagnostics; they do not select the best setting on the target data.
The defaults were declared before scoring. The point model was previously
selected using 2022, so the evaluation is not an untouched election holdout.

See [the generated evaluation](outputs/predictive-uncertainty-report.md) for
coverage, interval widths, misses and timings at each stage. The initial full
checks covered all party and bloc seat outcomes; the early ranges were often
wide. This suggests conservative ranges on this election, not proof that nominal
90% ranges calibrate in future elections. Another historical election pair and
better prior evidence for late/revision errors remain valuable. Probability
scores in the diagnostic files repeatedly score the same election outcome.

## Operation and reproducibility

The normal update makes 4,000 draws with a fixed seed. Use
`uv run update-report.py --draws 8000` for finer Monte Carlo resolution. The draw
count is recorded in report receipts and history identity. At 4,000 draws, an
estimated probability near 50% would have roughly 0.8 percentage points of Monte
Carlo standard error, before accounting for modelling error. Displayed percentages are rounded to whole numbers; endpoints
that would round to 0% or 100% appear as <1% or >99%. Lower-draw fixtures test the workflow only.

On 13 September, the 4,000-draw cached rehearsal rendered a complete report in
32.1 seconds (29.6 seconds for uncertainty). The updated partial-count mock's
uncertainty calculation took 32.3 seconds. Both are artificial examples; runtime
depends on reporting progress and these timings exclude a fresh network fetch.
All 122 tests passed, including the majority probability display and joint-seat
checks; 17 report tests also passed after checking percentage-rounding boundaries.

The frozen code lock includes `val2026/uncertainty.py` and
`val2026/forecast_uncertainty.py`. Configuration, diagnostics, seed and joint seat
draws accompany each prediction; `seat-draws.csv` provides the same seat draws
for inspection. A failed simulation preserves the previous complete forecast,
its intervals and its estimate timestamp. Older point-only history has no
invented retrospective intervals.

The pre-change Git checkpoint is `3239c6029a14ae2fa6bd655ef092f45094f64b64`.

Conceptual reference: [Stan posterior predictive sampling](https://mc-stan.org/docs/stan-users-guide/posterior-prediction.html).
