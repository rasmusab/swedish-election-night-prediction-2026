# Election-night preparation

## 1. Historical data

`scripts/prepare-history.py` reads the official 2022 final and preliminary district
workbooks, plus the original SCB 2018 workbook. It excludes blank and invalid
votes from party shares and calculates ÖVR as valid votes minus the eight named
parties. National valid totals reconcile to 6,476,725 (2018 final), 6,477,970
(2022 final workbook), and 6,445,298 (2022 preliminary registered-party votes).
The preliminary workbook's misleading valid-vote subtotal includes 3,481
unregistered-party votes. We instead use its explicit registered-party rows,
which reconcile with the archived machine-readable preliminary result.

The 2026 baseline uses actual 2022 final results and actual pre-election 2026
electorate data, never rehearsal vote counts. It contains 6,312 ordinary districts
and 290 municipality-level late-vote pools. Output: `data/processed/baseline-2026.csv`.

## 2. Changed districts and late votes

- Respect official comparability flags even if an old code is populated.
- Normalize the extra leading zero in the official Tranås Norra-Sommen source
  `006870101` to `06870101`; require every comparable source code to resolve.
- Comparable districts use the sum of mapped old party counts and electorate
  to derive shares and valid-vote rates. Some old codes feed multiple new
  districts; target electorate determines volume rather than copying old counts.
- For unmatched districts, use the municipality's historical districts left
  after removing sources already mapped to comparable new districts. If empty,
  fall back to municipality, then county averages. This is a proxy for the
  changed territory, not a claim of exact geographical reconstruction.
- The 2026 baseline has 1,253 districts using that residual municipality pool.
  The backtest has 2,118 such districts and two using the municipality average.
- Estimate volume as current electorate × historical valid-vote rate. The 2022
  electorate is read from final-result metadata but represents eligibility
  determined before voting. No target final vote counts enter prediction inputs.
- Aggregate collection districts by municipality: SCB 2018 collection district
  IDs repeat within municipalities. Late pools use prior municipal late-vote
  shares and prior late votes scaled by the municipality's electorate change.
- Keep late pools outstanding during election-night replay. The final outcome
  therefore remains uncertain even when all ordinary districts have reported.

## 3. Replay

`scripts/replay-baselines.py` reveals 2022 preliminary ordinary-district counts in steps
and scores against the actual final national result. Synthetic reporting orders
include random, small districts first, geographical delays, and large cities
late. Orders use only geography and known electorate. Development seeds are
0–2; later model evaluation reserves distinct seeds. These are sensitivity
tests on one historical election, not independent election-level validation.

An additional `archived_report_time` scenario uses the official surviving 2022
preliminary archive's report times. The archive includes two September 22
corrections, so this is not an exact reconstruction of first reporting on election
night. See `data/processed/2022-report-times-SOURCE.md`. The completed preliminary
counts likewise reflect later corrections; a replay cannot recover overwritten
original observations. The final workbook reconciles with the final archive at
every district/party. The preliminary workbook swaps district results for
01270007 and 01270008; we use corrected archived preliminary counts for all
districts. National totals are unchanged by that swap.

All missing observations are NaN and excluded from fitting. The feature table
has no outcome columns. `tests/test_replay.py` verifies that changing future
district results cannot change the prediction, late votes stay hidden, and
reported counts are retained.

Benchmarks include raw reported shares, unchanged historical forecasts for
remaining districts, national multiplicative swing, and county swing shrunk
toward the national swing. Score national party MAE, worst-party error, a stated
four-party bloc-margin error, vote-volume error, and missing-district error
separately for mapped and unmatched districts. This vote-model evaluation is
now supplemented by the seat backtest described below.

## 4. Collection and integration

`scripts/collect-results.py` defaults to one rehearsal check. Production must be selected
explicitly. It discovers archive names from the index, verifies checksums and
all official JSON signatures using the downloaded certificate, keeps immutable
snapshots, records source age and retries transient errors with backoff. Cached
good data survives unavailable feeds, checksum races and failed signatures.
Preliminary/final and rehearsal/production data are stored separately. Bounded
polling is available but no background collector has been started.

Live rehearsal collection succeeded with all three signatures verified and
6,626 district records. A production check returned HTTP 404 before election day
and was recorded as unavailable. The September 1 rehearsal source timestamp is
correctly flagged stale. `scripts/forecast-latest.py` reads verified preliminary snapshots
and joins them to the actual 2026 historical baseline, preserving observed
counts and separating incomplete late pools. Its full-count rehearsal test
reconciles exactly to all reported votes.

The forecasting model and retrospective iteration results are in MODEL-REPORT.md.


## Production report workflow (6 September 2026)

`refresh-inputs.py` redownloads the current official final 2022 workbook,
2022→2026 comparison and qualification-day electorate links. It validates them
before activating an immutable baseline snapshot. Production collection roster
metadata is bootstrapped only from a complete signed production snapshot; no
rehearsal metadata is needed. Until then ordinary results may be estimated while
all late pools remain incomplete, with observed votes preserved as lower bounds.

`election-night.py` is the interactive notebook and `update-report.py` executes
it in the uv interpreter, hides code and embeds charts in a single HTML file.
A process lock prevents overlap; failed exports leave the prior HTML intact.
Failed source/schema/model updates display the last successful estimate with
its original time and a failure state. The latest fetch attempt and source age
are always separate. `scripts/rehearse-report.py` exercised an eleven-case sequence and
exported waiting, early, partial, failure and complete report examples.

The operational runbook and unresolved production-feed gate are in
`ELECTION-NIGHT.md`. The earlier model performance conclusions remain unchanged.

## Seat projections (7 September 2026)

The frozen baseline and ordinary roster retain two-digit Riksdag constituency
codes and names. Every municipality maps to one constituency in the current
official geography, so all 290 late pools can be assigned without splitting
them. Ambiguous membership fails input preparation. A separate hashed
`fixed-seats-2026.csv` contains the official 310 fixed seats across 29
constituencies. Refresh activates that file together with the baseline, and
live forecasts check the feed’s `kretskod` against frozen membership.

`val2026/seats.py` independently implements party seat allocation from the
electoral rules. It handles thresholds, returned fixed seats, adjustment-seat
entitlements and their constituency placement. `scripts/backtest-seats.py`
validates national results in 2018 and 2022 and all constituency allocations in
2022 before evaluating the frozen forecast. See `outputs/seat-backtest-summary.md`.
The report now leads with party seats and the requested bloc comparison.
Approximate Bayesian joint simulations now provide 50%/90% party and bloc seat
ranges. The 2018→2022 uncertainty diagnostic scores 80 cases including component
removals and systematic-scale sensitivity. The ranges are assumption-based and
conservative in these checks; displayed majority probabilities are conditional
on the model assumptions. See
`UNCERTAINTY.md` and `outputs/predictive-uncertainty-report.md`.
