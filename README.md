# Swedish election 2026

[Election-night report](https://rasmusab.github.io/swedish-election-night-prediction-2026/) ·
[Source repository](https://github.com/rasmusab/swedish-election-night-prediction-2026)

See [intro.md](intro.md) for a high-level introduction to the data, statistical
model, uncertainty and Python workflow, followed by implementation details.

The local election-night workflow is ready to rehearse. See **ELECTION-NIGHT.md**
for the runbook, remaining live-feed checks, and publishing instructions.

```sh
uv run update-report.py --environment production
```

This pulls one preliminary snapshot, executes **election-night.py** (`# %%`
cells), fits the frozen model when data is available, and atomically writes
**index.html** at the project root with code hidden and charts embedded. A local
copy and supporting artifacts stay in `outputs/report-production/`. GitHub Pages
can serve `main` → `/(root)`; see [PUBLICATION.md](PUBLICATION.md).
Projected seats are the headline result, with party fixed/adjustment seats and
the V+S+MP+C versus M+L+KD+SD chart (175 seats for a majority).
It also shows 50%/90% seat ranges and simulated bloc distributions from 1,000
joint approximate Bayesian draws. These are assumption-based ranges, not
calibrated winner probabilities. See [UNCERTAINTY.md](UNCERTAINTY.md).
Run it again every ten minutes when wanted. No automatic monitor is running.

The [private hosted preview](https://swedish-election-2026-rasmus.rasmus-baath.chatgpt.site) is a published snapshot; local report generation does
not itself upload to Sites. The runbook distinguishes those steps.

To check this computer, the frozen inputs and the production connection without
fitting or publishing a forecast:

```sh
uv run scripts/check-readiness.py
```

The check writes `outputs/readiness/production/latest.md` and `.json`. Exit code
2 means an external check is pending (for example an unpublished production
index); 1 means a failure needing attention. `--environment rehearsal` checks
the test feed, and `--offline` checks local preparation without making requests.

To refresh authoritative background inputs before election night:

```sh
uv run refresh-inputs.py
uv run scripts/rehearse-report.py
uv run scripts/operational-rehearsal.py
```

The operational rehearsal uses the real public test feed, verifies signatures
and national/constituency/municipality totals, generates 1,000 uncertainty draws
and HTML, then tests a simulated connection failure and real recovery. It saves
evidence under `outputs/operational-rehearsal/` and does not replace the normal
production or mock reports. It respects the same server cooldown as normal updates.

The changed-district check is `uv run scripts/check-district-changes.py --draws 256`.
See [its report](outputs/district-changes/report.md): fallback districts have
larger local errors, while the existing constituency ranges remained conservative
in the 40-case historical diagnostic. The live model settings remain unchanged.

The baseline uses actual final 2022 results mapped to 2026 districts and the
2026 electorate. Production never reads the rehearsal roster. Immutable inputs
and hashes are selected by `data/inputs/latest-success.json`; `model-lock.json`
freezes the tested model implementation, options and dependency lockfile.
The frozen inputs also include the official 2026 fixed seats for all 29
constituencies. The tested allocator is included in the code lock.

For the filled, explicitly fictional preview:

```sh
uv run scripts/render-mock.py
```

Open `outputs/mock-report/index.html`. This isolated preview uses artificial
rehearsal votes and ten simulated updates; production data is unaffected.

The completed findings are in **MODEL-REPORT.md**, with an exportable comparison
chart at `outputs/backtest-performance.png` and `.svg`.

## Where the files live

Only three Python entry points remain at the top level:

| File | When to use it |
| --- | --- |
| `update-report.py` | Normal election-night command: collect, analyse and export HTML. |
| `election-night.py` | The same analysis as an interactive `# %%` notebook. |
| `refresh-inputs.py` | Refresh and freeze background data before election night. |

- `val2026/`: shared data, collection, model and report implementation.
- `scripts/`: historical exploration and optional rehearsal/diagnostic tools;
  see [scripts/README.md](scripts/README.md) for the distinction.
- `tests/`: automated checks.
- `data/` and `outputs/`: downloaded inputs and generated results.
- `old/`: the original archived project files.
- `site/`: the separately published static preview.

Downloaded data, generated outputs, the original archive and the separate preview
stay local and are ignored by Git. Source notes under `data/` and the frozen
`outputs/selected-model.json` configuration remain versioned. See
[PUBLICATION.md](PUBLICATION.md) for the public repository layout and setup of a
fresh checkout.

`uv sync --locked` installs this project in the existing environment, so imports
work both from the main notebook and when running a script in a subfolder.
The folder cleanup changes import paths, not the forecasting algorithms.

## Historical preparation and forecasting

The scripts use `# %%` cells and also run top to bottom. Original files under
`old/` remain unchanged. Dependencies are installed in `.venv` and locked by uv.

```sh
uv sync
uv run scripts/prepare-history.py
uv run scripts/replay-baselines.py
uv run scripts/backtest-model.py
uv run scripts/backtest-seats.py
uv run scripts/backtest-uncertainty.py --draws 256
uv run scripts/check-uncertainty.py
uv run scripts/check-uncertainty.py --late-shift
uv run scripts/report-results.py
uv run -m unittest discover -s tests -v
```

`scripts/backtest-model.py` defaults to the frozen selected model. To reproduce model
development: use `--phase development`, then `--phase iteration`, then
`--phase startup-check`, then `--phase evaluation`. The iteration/startup commands
write `outputs/selected-model.json`. Results and configuration for each phase
are saved separately. This is retrospective validation on one election; different
arrival seeds do not create independent election-level holdouts.

The seat backtest keeps that model unchanged, checks the allocator against
official 2018 and 2022 seats, and scores the rolling 2022 seat forecasts.
See [seat-backtest-summary.md](outputs/seat-backtest-summary.md) and
`outputs/seat-backtest-performance.png`. Seat estimates are point projections;
uncertainty is now evaluated separately; majority probabilities remain diagnostic.

`scripts/prepare-history.py` builds actual 2026 baselines and a 2018→2022 backtest table.
`PREPARATION.md` explains the changed-district and late-vote treatment. See
`data/SOURCES.md`, `data/processed/2022-report-times-SOURCE.md` and
`data/input-manifest.json` for provenance and file checksums.

## Collect and forecast the latest snapshot

```sh
# One rehearsal check and a forecast from the verified snapshot:
uv run scripts/collect-results.py --once
uv run scripts/forecast-latest.py

# Explicit production mode for election night:
uv run scripts/collect-results.py --environment production --once
uv run scripts/forecast-latest.py --environment production

# Optional finite collection run; no background service:
uv run scripts/collect-results.py --environment production --max-checks 60 --interval 60
```

The collector checks MD5 for changes and verifies every official JSON signature
with OpenSSL. It retains immutable snapshots, retries transient failures, honors
Retry-After, flags stale timestamps, and preserves the last successful data.
Preliminary and final feeds have separate storage (`--count final` downloads the
final-count feed). `scripts/forecast-latest.py` currently uses preliminary snapshots only.
Production being unavailable before polling day is handled explicitly.

The forecast reads only verified local data and needs at least two reporting
ordinary districts. All rehearsal output is labelled TEST DATA. Predictions
appear under `outputs/latest-rehearsal/` or `outputs/latest-production/`. A full
rehearsal count checks integration, not predictive accuracy. Partial late pools
retain known votes as lower bounds; their unfinished distribution remains a
model limitation. The point model is fast, but uncertainty is not calibrated for
winner probabilities. No monitor is running automatically.

## Original inspection scripts

Run from this folder:

```sh
uv sync
uv run scripts/inspect-2026.py
```

`scripts/inspect-2026.py` uses `# %%` cells for interactive Python editors, and also runs
top to bottom. Dependencies are locked in `uv.lock`; the environment is `.venv`.

The local `data` folder contains the official 2022–2026 district comparison
workbook and a **2026 Riksdag rehearsal snapshot**, with sources in
`data/SOURCES.md`. The script prints mapping counts, sample rows, reporting
metadata, district/party vote counts, vote-total checks, and geographic coverage.
All rehearsal vote counts are test data. Ordinary and late-vote collection
districts are kept separate. District and party codes retain leading zeros.

The script reads local files and does not modify the data or the old scripts.

## Try the rehearsal index

```sh
uv run scripts/check-rehearsal-index.py
```

This makes one public index request, discovers the preliminary Riksdag archive,
and downloads it if its checksum is not already cached. Run again to check for
changes and exercise the cache. Each distinct archive is kept under
`data/rehearsal-feed/snapshots/`, with timestamped check receipts under `checks/`.
`latest-check.json` records the most recent successful check. These downloads
do not replace the fixed input snapshot used by `scripts/inspect-2026.py`.

HTTP requests have timeouts. A 429 response stops with a retry message; the
script does not start a polling loop. Checksums verify consistency with the
index, not the published cryptographic signatures. Test data is explicitly
checked and labelled. The script never switches to production.
