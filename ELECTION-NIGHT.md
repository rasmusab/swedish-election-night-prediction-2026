# Election-night runbook

## Current state

Implemented and tested through 8 September 2026. The final 2022 baseline, 2026
geography and electorate have been refreshed from current links on
Valmyndigheten's official pages. There are 6,312 ordinary districts, 290
municipality late-vote pools, 8,046,725 eligible voters, and 1,253 ordinary
districts requiring a municipality-based historical fallback.
All 6,602 forecast units have a Riksdag constituency, including the late-vote
pools. The official 2026 fixed-seat distribution covers 29 constituencies and
310 seats; the allocator adds 39 adjustment seats.

The production `index.md5` still returns HTTP 404. This is the one live-source
validation gate that cannot be completed before publication. The report shows
“Waiting for results”; it never uses rehearsal votes as production results.

## File layout

The everyday entry points remain at the project root: `update-report.py`,
`election-night.py`, and `refresh-inputs.py`. Shared implementation is in
`val2026/`; historical exploration and optional tools are in `scripts/`.
See `scripts/README.md` for which files are exploratory and which are useful
for diagnostics. The main update command is unchanged.

During the cleanup, the model lock was updated for relocated files and the
editable project installation. The frozen model syntax was checked to be
identical apart from import paths, and numerical dependency versions were
unchanged. Verification and the previous lock are in `outputs/layout-cleanup/`.

## Before election day

From this project folder:

```sh
uv sync --locked
uv run refresh-inputs.py
uv run scripts/check-readiness.py
uv run python -m unittest discover -s tests -v
uv run scripts/rehearse-report.py
uv run scripts/operational-rehearsal.py
uv run scripts/backtest-seats.py --validate-only
uv run update-report.py --environment rehearsal
uv run update-report.py --environment production
```

The readiness check does not fit or publish a forecast. It tests the installed
runtime against `uv.lock`, all frozen model/input hashes, locally readable source
workbooks, OpenSSL, writable output/archive directories, current certificate
access, and feed alignment. Results are in `outputs/readiness/production/latest.md`
and `.json`. Use `--environment rehearsal` for the test feed, or `--offline` for
local checks without requests. Exit codes are 0 for passed checks (possibly
warnings), 1 for a failure, and 2 for pending external checks. A production 404
before polls close is pending; a timeout or failed signature is a failure.
After polls close an unavailable production index needs attention. A successful
download of old rehearsal data remains explicitly marked with a source-age warning.

Keep this Dropbox project, the active input snapshot, and `.venv` downloaded
locally. The readiness check reads the required input files rather than just
checking that their names exist. Run `uv sync --locked --offline` once before
the evening to confirm dependency installation needs no network. Keep the
computer powered and awake and have an alternative connection available.

Refresh background files before the evening; they are not downloaded on every
report update. `refresh-inputs.py` discovers current file links from the official
raw-data pages, checks district coverage, totals, mapping flags and historical
source references, then switches `data/inputs/latest-success.json` atomically.
Each snapshot retains its own source files, URLs, hashes and retrieval times.
A failed refresh leaves the active baseline untouched. `--offline` rebuilds from
existing downloaded workbooks and explicitly records that no fresh download
occurred.

The 8 September refresh retained input snapshot `8fb52b29e31aca073254ec28`;
the upstream background contents had not changed. Repeat the readiness check
and review any input refresh before Sunday evening, then hold the inputs fixed
during normal ten-minute updates. No background timer or future refresh job is
installed by these scripts.

The collector keeps the server's requested retry deadline across separate runs,
both feed environments, and readiness certificate probes;
re-running the report early skips network requests without extending that deadline.
If a cached signing certificate fails verification, it fetches the current
official certificate and verifies again once. An invalid replacement cannot
replace either the saved certificate or the last successful snapshot.

The model is the previously selected guarded share regression. It learns
changes from 2026 reporting ordinary districts using final 2022 district shares
and vote rates, 2026 electorate, and geographical effects. The 2018 data is used
only for the historical backtest. No Stan sampling or live model selection is
required. `model-lock.json` checks the model code, fallback implementation,
selected options and uv lockfile before forecasting. Do model development in a
separate copy; intentionally changing a locked file requires reviewing and
re-freezing it before live use.
The code lock also checks `val2026/seats.py`. Its addition did not change the
vote model, fitted options or numerical dependencies. The prior lock is retained
in `outputs/seat-implementation/previous-model-lock.json`.

When the production feed appears, run the production command and inspect
`outputs/latest-production/report-state.json`. The report automatically checks
that the election, count, parties, identifiers, municipality/type/constituency assignments
and valid-vote totals are consistent. A complete signed district list is checked
against the frozen ordinary roster and retained in
`data/collector/production/preliminary/roster.json`.
`roster_source.status: validated_full_feed` confirms roster validation.
If the first file omits unreported records, the status remains
`pending_full_feed`: ordinary forecasting can proceed, but no late pool is
classified as complete. An unexpected roster change fails visibly and retains
the previous estimate; investigate and refresh geography before resuming.

Every snapshot used for a forecast or readiness alignment check now requires
the district, mandate and municipality-summary JSON members from the same
signature-verified ZIP. Reported party votes and district counts must agree
nationally, in all 29 constituencies, and in all 290 municipalities. These
comparisons use counted votes only. They do not compare a final-result forecast
with the official partial count. File-generation times may differ by a few
seconds; matching timestamps are not assumed. Results are retained under
`aggregate_reconciliation` in the report state and readiness evidence.
An inconsistent archive fails the update and retains the last complete forecast,
including its uncertainty and original estimate time. Retry at the next update;
persistent disagreement requires investigation rather than bypassing validation.

The real incomplete-record convention remains to be checked against the first
production snapshot. The parser currently requires an explicit integer for each
of the eight report parties and ÖVR, plus reconciliation to valid votes. Missing
parties are never silently treated as zero. Unreported placeholders are excluded.
If the actual feed uses a different convention, the report stops updating its
estimate and gives the reason; the parser will need a documented adjustment.

## During election night

```sh
uv run update-report.py --environment production
```

Open the root `index.html` (also copied under `outputs/report-production/`). Repeat the command roughly every
ten minutes and reload the page. `election-night.py` can instead be run cell by
cell in an editor using `.venv`. It pulls data itself; the wrapper additionally
holds a lock through HTML export and executes a fresh notebook in the uv
interpreter. `--offline` renders cached data and visibly labels that no live
check was made.

The report leads with projected Riksdag seats for V+S+MP+C and M+L+KD+SD, a
stacked party chart with the 175-seat majority line, and party totals split into
fixed and adjustment seats. It also shows the bloc seat history, counted and
estimated eventual vote shares, valid votes reported and geographical coverage.
The report now adds central 50%/90% seat ranges, joint bloc seat distributions,
vote-share ranges and uncertainty shading in the history. These come from 4,000
approximate Bayesian simulations by default. Use `--draws 8000` for finer Monte
Carlo resolution. Bloc cards also show the fraction of simulations with at least
175 seats as an estimated majority probability, conditional on the model assumptions.
The point projection remains unchanged. The detailed fixed
and adjustment-seat breakdown can be expanded below the party table.
`outputs/latest-production/constituency-seats.csv` provides the projected
party allocation in each constituency. District CSVs retain constituency codes.
Expected vote coverage is a historical-volume measure, not the fraction of the
unknown final vote total. Fewer than two ordinary returns gives no estimate;
2–299 returns is explicitly labelled early. This threshold is an operational
label, not a guarantee of representativeness or accuracy.

Each report separates source update time, latest fetch attempt, estimate time,
and page generation time, all displayed in Swedish local time. Source age over
15 minutes is flagged even after a successful fetch. A newer generated page
does not mean newer votes. On failure, any retained forecast keeps its original
estimate timestamp. Model/schema failures and network failures are visible.
If notebook execution or HTML export itself fails, the command exits with an
error and the previously generated HTML remains intact; its old generation
time remains visible.

Repeated snapshots do not duplicate estimate history. Corrections replace
counts, including legitimate decreases. Older source timestamps cannot replace
a newer accepted snapshot. Complete district counts are preserved exactly;
partial late-vote counts remain in raw totals and bound the party predictions
from below. Retries are finite, Retry-After is respected, and the process lock
prevents two wrapper runs from overlapping. Do not run the standalone collector
concurrently with the wrapper; it is retained as a separate diagnostic tool.

## Counting phases and interpretation

Production estimates are suppressed before 20:00 Swedish time on 13 September.
After that the report always uses **preliminary** data. It labels the scheduled
pause after election night and the collection phase from Wednesday 16 September.
The 08:00 Monday phase-label boundary is our display convention, not an official
promise about the last Sunday-night update. The collector continues accepting
preliminary corrections in every phase.

The separate final recount starts Monday 14 September and overlaps with the
preliminary feed. It is never mixed into this model. `scripts/collect-results.py --count
final --environment production` can archive it separately for later analysis.
“Final-count feed” does not mean certified results until counting is complete.

The vote target is eventual national valid-vote share, including estimates of late
votes. Recount changes and late-vote composition remain model limitations.
At full preliminary completion the point forecast equals that preliminary
count. The uncertainty layer adds a separate assumed allowance for final-count
revisions; it does not consume the final feed. Historical tests cover one
election, not many independent elections. The ranges have not been calibrated
well enough for winner probabilities, which remain unpublished. Defaults were
conservative in the 2022 checks, particularly early in uneven reporting.
See [UNCERTAINTY.md](UNCERTAINTY.md) for the exact construction and prior scales.

Votes are aggregated across ordinary districts and municipal late pools within
each constituency. Seat allocation follows the modified Sainte-Laguë rules,
including the 4% national threshold, 12% local fixed-seat exception, returned
fixed seats and adjustment seats. Exact ties use a reproducible forecast lottery,
recorded in `prediction.json`; this is not an official lottery. The pooled ÖVR
category remains in the valid-vote denominator but is excluded from seats. This
assumes no individual party within ÖVR qualifies; modelling such a party would
require separate party data. Candidate shortages are outside this party-level
projection. Bloc labels are the requested groupings, not government predictions.

Official references:
- [Result-file specification and count phases](https://www.val.se/valresultat-och-statistik/statistik-och-data/teknisk-beskrivning-av-resultatfiler)
- [2026 raw geography/electorate data](https://www.val.se/valresultat-och-statistik/statistik-och-data/radata-val-2026)
- [Historical raw data](https://www.val.se/valresultat-och-statistik/statistik-och-data/radata-fran-val-2002-2022)
- [Seat allocation rules, chapter 14](https://www.riksdagen.se/sv/dokument-och-lagar/dokument/svensk-forfattningssamling/vallag-2005837_sfs-2005-837/)

## Publishing

The HTML is self-contained, with code hidden and PNG charts embedded. Only
the root `index.html` changes for routine website updates. Default production
runs replace it atomically after successful rendering. Rehearsal and custom
renders do not automatically replace it. Notebooks, data and receipts stay in
ignored folders.

For GitHub Pages, serve **main → /(root)** with the root `.nojekyll` file.
Commit and push the updated `index.html` to publish it; see
[PUBLICATION.md](PUBLICATION.md). The other tracked source files are also
accessible when serving the repository root. The public repository is
[rasmusab/swedish-election-night-prediction-2026](https://github.com/rasmusab/swedish-election-night-prediction-2026),
and Pages is configured at https://rasmusab.github.io/swedish-election-night-prediction-2026/.

A [private Sites preview](https://swedish-election-2026-rasmus.rasmus-baath.chatgpt.site) is published, with explicitly labelled rehearsal and failed-update examples. It is a fixed published snapshot,
not an automatic upload service. For updates to that preview, ask Codex to
republish the existing Site; `site/.openai/hosting.json` identifies it. The local
Python command does not hold reusable Sites credentials. The static source lives
under `site/dist/` and the hosted snapshot is built from that exact source.

The GitHub remote and Pages source were configured on 13 September 2026.
Verify the deployed homepage after each push. Local report generation does not
commit, push or schedule future updates; those remain separate actions.

## Verification performed

118 tests pass, including joint simulations, interval scoring, partial-pool
floors, final-versus-preliminary separation, seat thresholds, returned and adjustment seats,
constituency validation, feed signatures/tampering, checksum races, retry rules,
source rollback, downward corrections, future-result leakage, incomplete late
pools, model/input tampering, update locking and preservation of prior HTML on
execution failure. New checks cover signing-key rotation, shared retry deadlines,
resource-specific 404 handling, readiness failures, aggregate reconciliation and
changed-district scoring. The current test log is `outputs/readiness/tests.log`.

The allocator exactly reproduces national 2018 and 2022 party seats, and every
2022 constituency’s fixed, adjustment and total seats. The unchanged historical
forecast reaches a 0.5-seat mean party error at 10% of ordinary reports in the
archived order and exact final party totals from 75%. Early majority projections
can be wrong. This is one election, and archived timestamps can reflect later
corrections. The full 312-case evaluation and other reporting-order scenarios
are in `outputs/seat-backtest-summary.md`; fit plus allocation stayed below
0.02 seconds per selected-model point forecast in that run. The uncertainty
evaluation adds 80 cases with 256 draws each, including component removals and
half/double systematic-error scales. See `outputs/predictive-uncertainty-report.md`.
The 1,000-draw mock uncertainty calculation took about 8.3 seconds per update;
the vote fit and HTML export are additional, small costs.

The synthetic rehearsal drill passed eleven cases in 36.5 seconds and generated waiting, early,
partial, failed-update and complete HTML examples under
`outputs/rehearsal-drill/`. It simulates transport/signature responses in an
isolated temporary workspace; actual OpenSSL verification is tested separately
and succeeded for all three files in the real rehearsal archive. Rehearsal votes
are artificial and provide no evidence of 2026 prediction accuracy.

The fresh live operational rehearsal passed on 8 September: download, three real
OpenSSL signature checks, complete aggregate reconciliation, 1,000 simulations,
HTML, an injected network disconnection, and real recovery took 25.2 seconds.
Initial and recovered HTML runs took approximately 9.1 and 9.0 seconds; rendering
the retained estimate after the outage took 1.3 seconds. During the outage all
party/bloc seats, intervals, estimate time and history were preserved, and an
unchanged recovery archive did not duplicate history. Evidence and each HTML
stage are under `outputs/operational-rehearsal/`; its `summary.json` points to
the exact run. Test votes still date from 1 September and are correctly labelled
stale. Network failures can lengthen runs because retries are bounded but intentional.

The changed-district diagnostic (`scripts/check-district-changes.py`) completed
all 40 scenario/checkpoint cases with 256 unchanged joint simulations each.
Only unreported ordinary districts enter district scoring. Municipal fallback
had 2.78 pp vote-weighted mean absolute named-party share error versus 1.22 pp
for comparable districts. The 90% constituency party-share ranges covered 97.5%
of outcomes in the higher-fallback group and 99.0% in the lower-fallback group;
no scenario/checkpoint/group fell below 91.7%. From 50% ordinary reporting onward,
the maximum absolute party/constituency seat error was one seat. These dependent,
one-election diagnostics support retaining the tested model/settings for launch;
they do not establish calibration for 2026. See `outputs/district-changes/report.md`
for scoring definitions, remaining misses, CSVs and the 2026 exposure table.
