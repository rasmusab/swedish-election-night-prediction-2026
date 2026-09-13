# Public repository and local files

The public source tree contains the Python program, supporting scripts, tests,
documentation, dependency locks and frozen model settings. Local working data
remain in their existing folders so election-night commands keep working.

| Folder | Contents | Git policy |
| --- | --- | --- |
| `data/` | Downloads, prepared inputs, live archives, caches and receipts | Ignored except `SOURCES.md` and `processed/2022-report-times-SOURCE.md` |
| `outputs/` | HTML reports, executed notebooks, simulations, charts and logs | Ignored except `selected-model.json`, required by the frozen model |
| `old/` | Original project files and the previous local Git history | Ignored |
| `site/` | Earlier static preview, with its own repository | Ignored |
| `.venv/`, `.uv-cache/` | Local Python environment and package cache | Ignored |

The fresh repository was initialized on 13 September 2026. The previous Git
directory is saved locally at `old/git-history-before-publication/`; it is not
part of the new history. Original working data have not been deleted.

## Starting from a fresh checkout

The checkout contains source and configuration, not the downloaded inputs.
To initialize the live workflow from the official sources:

```sh
uv sync --locked
uv run refresh-inputs.py
uv run scripts/check-readiness.py
uv run update-report.py --environment production
```

This requires network access and the official source workbooks to remain
available. The readiness check also requires OpenSSL. Its exit code 2 means
an external check remains pending; 1 means a failure requiring attention.
Existing installations already have frozen inputs and do not need to refresh
them merely because Git was reinitialized.

Historical backtests and some integration tests additionally require the local
historical downloads and original 2018 files documented in `data/SOURCES.md`.
They are not all supplied or reconstructed by the live input refresh command.
Links in the research documentation to generated files under `outputs/` refer
to local artifacts; those files will not be present in a source-only checkout.

## Publishing the website

The intended public website artifact is only
`outputs/report-production/index.html`, copied into a dedicated publishing
branch as `index.html`. The executed notebook, caches, logs and archived Git
history are not website assets. A publishing branch and GitHub Pages deployment
have not yet been configured.

Ignoring a folder keeps it out of normal commits; force-adding its files would
override that exclusion. Preserve the exclusions when adding a GitHub remote.
