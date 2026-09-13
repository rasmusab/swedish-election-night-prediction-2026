# How the election-night forecast works

This project estimates the eventual distribution of Swedish Riksdag seats while votes are still being counted. It combines the results already reported in 2026 with estimates for districts that have not yet reported, then translates the combined votes into party and bloc seats.

The starting idea is that districts have persistent political characteristics. A district that voted strongly for a party in 2022 provides a useful baseline for 2026, but the national and local patterns of change are unknown. As results arrive, we learn those changes from reporting districts and apply them to the districts still outstanding.

The implementation uses a fast, regularized regression for the point forecast and approximate Bayesian simulations for uncertainty. Each update can run locally in Python and produce a self-contained HTML report. The description below reflects the configuration checked on **8 September 2026**.

## 1. The idea in one example

Suppose the first reporting districts show that a party is doing better than in 2022, particularly in districts where another party previously had a strong vote. We can use that pattern to adjust the historical expectations for similar districts that have not reported.

Simply extrapolating the current national vote share would miss this distinction. Early reporting districts are not a random sample of Sweden: their size, location and political composition can differ from the country as a whole. The regression tries to account for these differences using characteristics known before the election.

At any moment, the projected vote total for a party is conceptually:

```text
projected votes = votes already reported + estimated votes still outstanding
```

This is done within each Riksdag constituency, because constituency votes matter for seat allocation. The report's headline outcome is the number of seats for each party and for these two specified groupings:

- **V + S + MP + C**
- **M + L + KD + SD**

There are 349 seats, so 175 constitutes a majority. The groupings are definitions used in this report; the statistical model does not predict coalition negotiations or government formation.

## 2. What data do we use?

Three inputs support the live forecast, with additional data for historical testing and rehearsals:

| Data | Role |
| --- | --- |
| Final 2022 district results | Historical party shares and expected valid-vote volumes for the 2026 forecast. |
| Official 2026 geography, district-comparison mapping, electorate and fixed-seat allocation | Describe the units being forecast and how they contribute to seats. |
| Preliminary 2026 district results, arriving during counting | Supply the observed votes and the training observations for each new regression fit. |
| 2018 results and archived preliminary/final 2022 results | Historical development and testing: pretend 2022 is arriving, with 2018 as its background. |
| Official artificial rehearsal results | Test downloading, validation, forecasting and HTML generation. They provide no evidence about the likely 2026 result. |

We currently use neither opinion polls nor exit polls, individual voter data, or external demographic predictors. District size, historical voting and geography provide the explanatory information.

### Districts, municipalities and constituencies

The geography has several levels that serve different purposes:

- **6,312 ordinary polling districts** (*valdistrikt*): the main forecasting and regression-training units.
- **290 municipalities** (*kommuner*): local regression effects and aggregation of late collection votes.
- **21 counties** (*län*): broader geographical regression effects.
- **29 Riksdag constituencies** (*valkretsar*): the units used in seat allocation.

Counties and Riksdag constituencies are not interchangeable. The model has county and municipality effects; the seat calculation uses the actual constituency assignments.

We also maintain one late-vote pool per municipality, making **6,602 model rows** in total. A pool can correspond to multiple official collection districts. These are model aggregation units, so the model row count need not equal the number of records in a result archive.

### Mapping 2022 into 2026 geography

District boundaries changed between elections. We use Valmyndigheten's official comparability assessment rather than assuming an unchanged district code means unchanged geography:

| Official comparison for 2026 districts | Number |
| --- | ---: |
| Comparable with one 2022 district | 5,024 |
| Comparable with multiple 2022 districts | 35 |
| Not directly comparable | 1,253 |

For comparable districts, we aggregate the mapped historical counts and electorate, then calculate historical party shares and valid-vote rates. The current electorate determines the expected 2026 volume. Thus, when old districts are shared between mappings, their whole vote totals are not simply copied into every successor.

For non-comparable districts, the preferred baseline comes from the municipality's historical districts not already used in comparable mappings. The code also supports broader municipality or county fallbacks if needed. All 1,253 non-comparable ordinary districts in the current 2026 inputs use the residual-municipality baseline. They cover about **19.6% of the electorate**.

This preserves useful municipal information, but it does not reconstruct exactly how people inside each new boundary voted in 2022. The model includes an indicator for whether a district needed a fallback.

### Receiving and validating results

Valmyndigheten's public interface supplies an index and ZIP archives containing signed JSON. Each update checks the index, obtains the relevant preliminary Riksdag archive, verifies its checksum and signatures, and validates its contents. No API key is required for this public file interface.

Before forecasting, district party totals must reconcile with the official national, constituency and municipality aggregates from the same archive. The district roster must also agree with the frozen geography. An unreported placeholder is treated as missing; it is not a district with zero votes. A reported district must have explicit counts for all modelled party categories.

Background files and model settings are frozen and identified by hashes. They are not downloaded or retuned on every update. Current-source references and download provenance are retained with each input snapshot; the external interface is described in [Valmyndigheten's technical documentation](https://www.val.se/valresultat-och-statistik/statistik-och-data/teknisk-beskrivning-av-resultatfiler).

## 3. What is the statistical model?

The selected model is called **`guarded_share_ridge`**. It is a weighted ridge regression for changes in party shares and valid-vote volume. It has a Gaussian-prior interpretation, but fitting the point estimate requires only linear algebra.

An important distinction: **we do not train a 2022 outcome model once and then apply it unchanged to 2026.** Historical data supplies baselines, and historical experiments helped choose the model form. During election night, the coefficients are fitted afresh to the ordinary 2026 districts that have reported so far. The chosen predictors and regularization settings stay fixed.

### Responses: changes from the historical baseline

There are nine party categories: M, C, L, KD, MP, S, V, SD and ÖVR. ÖVR pools other valid party votes; blank and invalid ballots are outside the party-share denominator.

For each reported ordinary district, the regression has ten responses:

1. Nine differences between its current party shares and its historical baseline shares.
2. The logarithm of its current valid-vote total divided by its historical expected valid-vote total.

For district `i` and party `p`, the share response is:

```text
share_change[i, p] = smoothed_current_share[i, p] − historical_share[i, p]
volume_change[i]   = log(current_valid_votes[i] / expected_valid_votes[i])
```

Shares are smoothed by adding half a vote to each category before dividing by the adjusted total. This stabilizes small and zero counts; actual reported counts are retained unchanged in the final aggregation.

The volume response measures **valid party votes**, not official turnout including invalid ballots. Its historical baseline is approximately the district's current electorate multiplied by its historical valid-vote rate.

The active model regresses changes in ordinary shares. The code also contains a centered-log-ratio variant explored during development, but that is not the selected live configuration. Neither the active regression nor its uncertainty layer is a multinomial model fitted in Stan.

### Predictors and geographical pooling

Each response uses the same predictors:

- All nine historical party shares.
- Log electorate and historical valid-vote rate.
- The historical-fallback indicator.
- An intercept, county indicators and municipality indicators.

The numerical covariates are standardized using the ordinary districts' pre-election features. No unreported 2026 outcome is needed for this preprocessing.

Regularization shrinks coefficients towards zero. For a change model, zero means no adjustment to the historical baseline. A municipality with few reports therefore cannot freely estimate a large local change; it draws more of its prediction from the broader fitted pattern. This is a form of partial pooling with fixed shrinkage strengths.

The model captures associations useful for prediction. A coefficient involving a party's previous share is not an estimate of individual voters switching between parties.

### The fitting calculation

Let `X` be the reported districts' design matrix, `Y` their ten response columns, and `W` a diagonal matrix of observation weights. The coefficient estimate is:

```text
B_hat = solve(Xᵀ W X + Λ, Xᵀ W Y)
```

`Λ` is a diagonal penalty matrix. Current defaults are:

| Coefficient group | Penalty |
| --- | ---: |
| Intercept | 0.001 |
| Standardized numerical predictors | 20 |
| County effects | 10 |
| Municipality effects | 10 |

The observation weight is `clip(sqrt(valid_votes / 1000), 0.3, 2.0)`. Larger reporting districts get more weight, but their influence grows much less than proportionally to vote count. These are modelling weights, not a survey sampling design.

The solution can be interpreted as a conditional Gaussian-prior maximum-a-posteriori estimate. Shrinkage strengths are fixed rather than inferred in a full hierarchical Bayesian fit. NumPy and SciPy perform the matrix operations, with a Cholesky solve for the coefficient system. There is no MCMC convergence or warm-up stage at each update.

### Producing the point forecast

For each outstanding district, the fitted share changes are added to its historical shares. The resulting shares are floored at a small positive value and normalized to sum to one. Expected volume is multiplied by the exponentiated fitted log-volume change; this change is clipped to `[-1, 1]` before exponentiation. The ordinary-district regression volume is capped at its frozen electorate.

Shares multiplied by volume give predicted party votes. Completed districts are then replaced by their actual reported counts. This ensures the forecast improves by replacing estimates with observations as the count progresses.

Startup receives additional protection. With fewer than two ordinary reports there is no numerical estimate. Through 100 reports, the point forecast averages the regression and a simpler county-swing forecast equally. The regression's weight then rises linearly to 100% at 300 reports. The simpler forecast adjusts historical shares using observed national and county party ratios, shrinking sparse county information towards the national change.

The report labels estimates with fewer than 300 ordinary returns as early. That label and the blend limit extrapolation; they cannot make the first reporting districts representative.

## 4. How do late votes enter?

Late collection votes remain part of the target even when every ordinary district has reported. Historical late votes can have a different party composition from ordinary votes, so each municipal pool has its own historical late-vote shares and expected volume.

Late pools do not train the ordinary-district regression. To predict changes for them, the design matrix uses electorate-weighted average ordinary-district covariates from their municipality, while retaining the pool's own historical shares and volume as its starting point.

As collection-district returns arrive, their observed votes become a fixed lower bound. A pool becomes fully observed only once the verified roster is complete and all its expected collection records have reported. If the roster is still incomplete, seeing all currently listed records does not establish completion.

The live workflow follows the preliminary counting stream. It does not combine it with the separate final-count stream. The point forecast has no separate fitted recount correction; the uncertainty model adds an allowance for differences between the completed preliminary count and the eventual final result.

## 5. Turning votes into seats

Predicted ordinary and late votes are summed within all 29 constituencies. The allocator then produces one complete party allocation of 349 seats, using the frozen 2026 distribution of 310 fixed seats and the adjustment-seat procedure.

The implementation includes the national 4% threshold, the 12% constituency exception for fixed seats, the modified Sainte-Laguë procedure, returned fixed seats and adjustment seats. It works with the predicted vote weights directly, avoiding an unnecessary rounding of forecasts into invented integer vote counts. Exact quotient ties use a recorded, reproducible forecast lottery.

ÖVR stays in the valid-vote denominator but receives no seats as a pooled category. This assumes that no party hidden inside ÖVR needs an individual seat forecast. Candidate selection, personal votes and candidate shortages are outside the model's scope.

National share alone would be insufficient to implement all of this faithfully. Keeping constituency-level predictions also allows every uncertainty simulation to pass through the same allocation procedure.

## 6. What does uncertainty mean here?

The report shows a point projection alongside central **50% and 90% predictive ranges**. These describe simulated eventual outcomes under the model and its assumptions. They are not guaranteed bounds, and a 90% label has not been calibrated using a large collection of independent elections.

The uncertainty layer constructs complete, dependent election outcomes. Parties compete for the same votes and seats; districts can share an uncertain national or local swing. Treating every district and party as independent would allow too much uncertainty to cancel when aggregating.

There are five components:

| Component | What it represents |
| --- | --- |
| Regression-coefficient uncertainty | We have only a limited number and range of reporting districts from which to learn the swing. |
| District residual variation | Districts differ even after conditioning on the fitted predictors. |
| A common reporting-selection discrepancy | Outstanding ordinary districts may systematically differ from reporting districts in ways the regression misses. |
| Late-vote uncertainty | Late pools have local variation and can share a different national change or volume shift. |
| Final-count revisions | The eventual final result can differ from the completed preliminary count. |

### Approximate Bayesian draws around the regression

Conditional on the fixed penalties and an estimated response covariance `Σ`, the coefficient covariance has the matrix-normal form:

```text
Cov(vec(B)) ≈ Σ ⊗ inverse(Xᵀ W X + Λ)
```

Here `Σ` describes dependence among the nine share changes and the log-volume change. Each simulated election receives one shared coefficient perturbation, which influences all outstanding districts through their predictors. County and municipality coefficient uncertainty is included in this draw; a second independent set of geographical effects is not added.

The response covariance combines weighted residuals from reported ordinary districts with a stabilizing prior equivalent to 40 residual degrees of freedom. This prevents a handful of early reports from implying negligible uncertainty. The calculation uses a ridge-adjusted residual degrees-of-freedom estimate.

For historical national shares `p`, the prior party-share covariance is `0.05² × (diag(p) − p pᵀ)`, and the prior log-volume standard deviation is `0.15`. The party covariance respects competition between shares; `0.05` is a scale in this formula, not a five-percentage-point standard deviation for every party.

The response covariance is then held fixed during simulation. We do not sample a posterior over all covariance and shrinkage parameters. This, together with the additional assumed error components, is why the procedure is described as **approximate, conditional Bayesian simulation**.

### Extra protection against shared errors

The common discrepancy for outstanding ordinary districts uses half the fitted residual standard deviation, multiplied by a reporting-imbalance factor between 0.25 and 1. That factor measures imbalance in county, municipality population and district size using known geography and the membership of the reporting set. It does not observe how the missing districts will vote.

Late pools receive both local residual variation and a shared national late effect. The shared late log-volume standard deviation has a floor of 0.20. Partially counted pools retain their known votes, with a one-sided allowance for further votes if those known counts already exhaust the point estimate.

Finally, after aggregating the simulated preliminary votes, a small common revision layer perturbs constituency shares and volumes. Its default share scale is 0.05 percentage points and its log-volume scale is 0.001. These are explicit sensitivity assumptions, not estimated upper bounds on recount differences.

Thus, completed reported votes are fixed in the **preliminary simulation layer**. The separate **final-result layer** can revise them. Uncertainty about final results can survive a fully reported preliminary count. The present revision treatment would need reconsideration when handing over to certified final results.

Simulated shares are kept nonnegative and normalized, and outstanding ordinary volumes are capped at the electorate. Verified observed votes are never truncated to that frozen figure. These constraints and the one-sided late allowance can move the simulation mean away from the point projection; diagnostics record the difference rather than forcing the draws back to the headline estimate. The startup blend also retains full regression covariance as a conservative approximation, rather than deriving a full posterior for the blended estimator.

### From joint draws to party and bloc ranges

By default, an update produces **1,000 joint election draws**. Each draw is aggregated by constituency and receives a complete seat allocation. Bloc seats are summed within that draw, and only then are their quantiles calculated. Adding the endpoints of individual party intervals would not give a valid bloc interval.

A party near 4% can have two separated groups of seat outcomes: no seats below the threshold and a substantial allocation above it. A single interval can hide this structure, which is why the project retains seat histograms and individual simulated allocations. The headline allocation comes from the point vote forecast; separate party medians need not add up to 349.

Majority probabilities are calculated for diagnostics but are not displayed as calibrated winning probabilities in the HTML. More draws reduce Monte Carlo noise, not model misspecification. With 1,000 draws, a simulated probability near 50% has about 1.6 percentage points of Monte Carlo standard error alone.

## 7. How do we run it in Python?

Run the following from the project folder. Dependencies are managed by `uv` and pinned in `uv.lock`; the project uses Python 3.14 or newer.

```sh
uv sync --locked
uv run scripts/check-readiness.py
uv run update-report.py --environment production
```

The readiness check does not fit or publish a forecast. It verifies the local setup, frozen inputs and feed. Exit code 2 means a check remains pending, such as an unpublished production index; exit code 1 means a failure needing attention.

The normal update collects one snapshot, validates it, executes the analysis and writes the root `index.html` for GitHub Pages, retaining a local copy in `outputs/report-production/index.html`. Run it again roughly every ten minutes and reload the file; publishing requires committing and pushing the updated root HTML as described in `PUBLICATION.md`. No background schedule is started. Production estimates are suppressed before polls close.

The analysis itself is `election-night.py`, a Python file organized into `# %%` cells. It can be run interactively in the project's environment. `update-report.py` executes those cells in a fresh notebook kernel, hides their code in the exported HTML, embeds the graphs, and replaces the previous HTML atomically after a successful export.

For a real-feed integration test or a cached update:

```sh
uv run update-report.py --environment rehearsal
uv run update-report.py --environment production --offline
```

The rehearsal uses artificial votes. Offline mode uses stored collection state and explicitly records that no live check occurred; it does not turn a previously failed collection into a successful one.

The Python entry point underlying the notebook is:

```python
from val2026 import ROOT
from val2026.locking import update_lock
from val2026.election_report import run_cycle

with update_lock(ROOT):
    state = run_cycle(ROOT, "rehearsal", simulation_draws=1000)

# A failed update may retain an older estimate. Check its health and timestamps.
print(state["health"], state.get("source_updated_at"))
print(state.get("blocs", []))
```

This collects, fits when possible and saves analysis state; it does not itself export the notebook to HTML. Most routine use should go through `update-report.py`. The regression implementation is in `val2026/forecast_model.py`, joint vote simulations in `val2026/uncertainty.py`, and simulated seat summaries in `val2026/forecast_uncertainty.py`.

The numerical regression fit has taken milliseconds in the tested runs. A full HTML render with 1,000 draws took about nine seconds in the 8 September operational rehearsal; a fresh download, initial report, simulated outage report and recovered report together took about 25 seconds. These are measured examples, not worst-case network guarantees, but they leave considerable room inside a ten-minute update cycle.

## 8. What have we learned from testing?

The historical replay uses 2018 as background and reveals preliminary 2022 ordinary-district results progressively. Final 2022 outcomes enter validation and scoring. Five arrival orders include surviving archived reporting timestamps and four synthetic scenarios, such as small districts arriving first or large cities arriving late. Late pools remain outstanding even at 100% ordinary reporting.

The main diagnostic commands are:

```sh
uv run scripts/backtest-seats.py
uv run scripts/backtest-uncertainty.py --draws 256
uv run scripts/check-district-changes.py --draws 256
uv run scripts/operational-rehearsal.py
```

The allocator reproduces historical national party seats for 2018 and 2022, and the 2022 constituency fixed, adjustment and total allocations. Separate transport and workflow tests exercise missing data, corrections, signature failures, mismatched aggregates, outages and recovery.

The changed-district diagnostic found larger errors for unreported municipal-fallback districts: 2.78 percentage points versus 1.22 for comparable districts, using vote-weighted mean absolute share error over eight named parties. Some local errors cancel in larger aggregates. Existing 90% constituency party-share ranges covered 97.5% of outcomes in the higher-fallback group and 99.0% in the lower-fallback group across the tested scenarios and stages. This supported retaining the current configuration for launch.

These are repeated views of **one historical election**. The point model was also selected using that election, and archived timestamps can reflect later corrections rather than first arrival. Neither many reporting orders nor many simulation draws create independent elections. The checks are useful for detecting failures and understanding sensitivity, but they do not establish reliable 2026 coverage probabilities.

The central statistical assumption remains that the changes learned from reporting districts transfer sufficiently well to outstanding ones after accounting for the available predictors. Unusual local swings, systematically delayed areas, imperfect boundary mappings, late voting and revisions can challenge that assumption. The extra uncertainty components make some of those risks explicit; their magnitudes remain modelling judgments.

## 9. Further reading in this project

- `ELECTION-NIGHT.md`: practical preparation, commands, failure handling and the remaining first-production-file checks.
- `UNCERTAINTY.md`: the simulation construction and its assumptions in more detail.
- `MODEL-REPORT.md`: historical regression development and comparisons.
- `outputs/district-changes/report.md`: comparable/fallback diagnostics, charts and constituency results.
- `outputs/predictive-uncertainty-report.md`: interval performance and sensitivity experiments.
- `scripts/README.md`: which optional scripts support operation, testing or earlier exploration.

The ordinary workflow is intentionally small: verified result files go in; a fitted regression, joint vote simulations and seat allocation produce the report. The historical experiments and operational drills support that workflow without becoming additional steps in each election-night update.
