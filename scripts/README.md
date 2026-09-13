# Optional scripts

These are not required for a routine election-night update. Run them from the
project folder with `uv run scripts/<name>.py`. They also retain `# %%` cells
for interactive use in the project's `.venv` environment.

## Rehearsal, diagnostics and publishing preparation

| Script | Purpose |
| --- | --- |
| `check-readiness.py` | Check the local runtime, frozen inputs, certificate access, feed and district alignment without fitting or publishing. |
| `rehearse-report.py` | Exercise partial results, corrections and failures in an isolated rehearsal. |
| `operational-rehearsal.py` | Fresh real rehearsal download, signatures, forecast and HTML, followed by a simulated disconnection and real recovery. |
| `render-mock.py` | Regenerate the one-off filled HTML preview, with artificial votes, seat charts and ten simulated updates. |
| `collect-results.py` | Collect and verify snapshots without running the model or HTML export. |
| `forecast-latest.py` | Inspect a forecast from the verified local cache. |
| `stage-preview.py` | Copy generated HTML into the existing private preview checkout; does not publish. |

## Historical exploration and model development

| Script | Purpose |
| --- | --- |
| `inspect-2026.py` | Inspect the downloaded 2026 mapping and artificial rehearsal data. |
| `check-rehearsal-index.py` | Original exploration of the public rehearsal index. |
| `prepare-history.py` | Rebuild historical features and the 2018→2022 backtest table. |
| `replay-baselines.py` | Compare simple historical/swing baselines under simulated reporting orders. |
| `backtest-model.py` | Evaluate or develop the regression using the historical backtest. |
| `backtest-seats.py` | Validate official historical seats and score the unchanged model’s rolling 2022 seat projections. |
| `backtest-uncertainty.py` | Evaluate joint approximate Bayesian vote/seat ranges, component removals and prior-scale sensitivity on 2022. |
| `check-district-changes.py` | Score unreported comparable/fallback districts separately, constituency seats and uncertainty coverage; leaves model settings unchanged. |
| `check-uncertainty.py` | Earlier national-share uncertainty experiment; retained as a diagnostic comparison. |
| `report-results.py` | Regenerate the historical model report and comparison chart. |

`uncertainty.py` and `rehearsal_scenarios.py` are supporting modules for these
optional scripts and tests. The live model does not import them.

Development phases can change the selected model configuration; use a separate
copy for further model development while preparing the frozen live workflow.
