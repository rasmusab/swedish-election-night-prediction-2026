# %%
"""Real public rehearsal → signed archive → forecast → HTML → outage → recovery.

Run with the project's uv environment:
    uv run scripts/operational-rehearsal.py

Each run starts with an empty collector cache in a temporary workspace. Results
are saved under outputs/operational-rehearsal/runs/<UTC timestamp>. Artificial
rehearsal votes test the machinery, not the accuracy of the 2026 predictions.
No production report, mock report, network settings or background jobs change.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter
from urllib.error import URLError

from val2026 import ROOT
from val2026.forecast_uncertainty import DEFAULT_DRAWS
from val2026.election_feed import (Collector, HttpClient, FetchError, RetryDeferred,
                                  atomic_write, json_bytes, record_retry_deadline)
from scripts.rehearsal_scenarios import copy_inputs

DRAW_COUNT = DEFAULT_DRAWS
COOLDOWN_PATH = Path('data/collector/retry-not-before.json')
PRESERVED_FIELDS = ('parties', 'seat_allocation', 'blocs', 'uncertainty',
                    'estimate_generated_at_utc', 'history', 'aggregate_reconciliation')


# %% Shared server cooldown: an isolated test must not bypass other callers.
def read_cooldown(root):
    path = root / COOLDOWN_PATH
    if not path.exists():
        return None
    saved = json.loads(path.read_bytes())
    deadline = datetime.fromisoformat(saved['retry_not_before_utc'])
    if deadline.tzinfo is None:
        raise ValueError('The shared server retry deadline must include a timezone')
    return RetryDeferred(deadline, now=datetime.now(timezone.utc),
                         status=saved.get('http_status'), url=saved.get('failed_url'))


def share_cooldown(source_root, destination_root):
    error = read_cooldown(source_root)
    if error is not None:
        record_retry_deadline(destination_root, error)


class SharedCooldownClient(HttpClient):
    """Check the real project's host cooldown before every live HTTP request."""

    def __init__(self, shared_root, **kwargs):
        super().__init__(**kwargs)
        self.shared_root = shared_root

    def get(self, url, max_bytes):
        deferred = read_cooldown(self.shared_root)
        if deferred is not None and deferred.deadline > datetime.now(timezone.utc):
            raise deferred
        try:
            return super().get(url, max_bytes)
        except FetchError as error:
            record_retry_deadline(self.shared_root, error)
            raise


class RehearsalUnavailable(RuntimeError):
    """The live prerequisite is pending; never replace it with cached data."""


def check_feed(workspace, shared_root, *, disconnected=False):
    share_cooldown(shared_root, workspace)
    if disconnected:
        def offline_opener(request, **kwargs):
            raise URLError('Injected operational rehearsal disconnection; OS network unchanged')
        client = HttpClient(opener=offline_opener, max_attempts=1, timeout=20)
    else:
        client = SharedCooldownClient(shared_root, max_attempts=2, timeout=20, max_retry_wait=10)
    try:
        # Always use the real default OpenSSL verifier; only the disconnection
        # request transport is replaced, and that request cannot return data.
        return Collector(workspace, environment='rehearsal', client=client).check()
    finally:
        share_cooldown(workspace, shared_root)


# %% Full notebook execution and HTML export with a just-collected local feed.
def exporter_module():
    spec = importlib.util.spec_from_file_location('operational_report_exporter', ROOT / 'update-report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_stage(exporter, workspace, destination, stage, check):
    folder = destination / stage
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write(folder / 'collector-receipt.json', json_bytes(check))
    # Collect explicitly above so the kernel does not make a second fetch.
    # The ordinary exporter still performs every parse/model/chart/HTML step.
    rendered = exporter.render(workspace, 'rehearsal', offline=True,
                               output=folder / 'index.html', draws=DRAW_COUNT)
    state = json.loads((workspace / 'outputs/latest-rehearsal/report-state.json').read_bytes())
    atomic_write(folder / 'report-state.json', json_bytes(state))
    if state.get('test') is not True or state.get('environment') != 'rehearsal':
        raise AssertionError('Operational rehearsal report has the wrong environment')
    return state, {'collector': check, 'render': rendered,
                   'health': state['health'], 'source_updated_at': state.get('source_updated_at'),
                   'estimate_generated_at_utc': state.get('estimate_generated_at_utc'),
                   'aggregate_reconciliation': state.get('aggregate_reconciliation')}


def require_live_success(receipt, stage):
    if receipt.get('status') != 'ok':
        if receipt.get('http_status') in (404, 429) or receipt.get('network_request_skipped'):
            raise RehearsalUnavailable(f"{stage}: {receipt.get('error', 'live rehearsal unavailable')}")
        raise RuntimeError(f"{stage}: {receipt.get('error', 'live collection failed')}")
    signature = receipt.get('signature', {})
    if signature.get('status') != 'verified' or len(signature.get('json_files_verified', [])) != 3:
        raise AssertionError(f'{stage}: all three official signatures must be verified')


def require_forecast(state):
    if state.get('health') == 'update_failed':
        raise RuntimeError(state.get('error', 'Forecast failed'))
    if state.get('uncertainty', {}).get('draws') != DRAW_COUNT:
        raise AssertionError(f'The real rehearsal did not generate {DRAW_COUNT:,} uncertainty simulations')
    if state.get('aggregate_reconciliation', {}).get('status') != 'reconciled':
        raise AssertionError('The real rehearsal did not reconcile official aggregates')
    if sum(p['forecast_seats'] for p in state['parties']) != 349:
        raise AssertionError('Forecast seats do not sum to 349')


# %% Run the drill and retain evidence even when the live prerequisite fails.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/operational-rehearsal')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    destination = output / 'runs' / started_at.strftime('%Y%m%dT%H%M%S%fZ')
    destination.mkdir(parents=True)
    summary = {'status': 'running', 'started_at_utc': started_at.isoformat(),
               'environment': 'rehearsal', 'simulation_draws': DRAW_COUNT,
               'run_directory': str(destination), 'stages': {}, 'checks': {},
               'scope': 'Real public rehearsal download, signatures, aggregates, model, HTML, simulated outage and recovery. '
                        'Artificial votes do not validate 2026 predictive accuracy.',
               'rendering': 'Each notebook uses the preceding collection receipt without a second network fetch.'}
    with tempfile.TemporaryDirectory(prefix='val2026-operational-') as temporary:
        workspace = Path(temporary)
        try:
            copy_inputs(workspace)
            manifest = json.loads((workspace / 'data/inputs/latest-success.json').read_bytes())
            summary['input_snapshot_id'] = manifest['snapshot_id']
            summary['model_lock_id'] = json.loads((workspace / 'model-lock.json').read_bytes())['lock_id']
            exporter = exporter_module()
            if (workspace / 'data/collector/rehearsal/preliminary').exists():
                raise AssertionError('Initial live fetch must start without a collector cache')

            initial_check = check_feed(workspace, ROOT)
            summary['stages']['initial'] = {'collector': initial_check}
            require_live_success(initial_check, 'Initial fetch')
            if initial_check.get('archive_downloaded') is not True:
                raise AssertionError('The first rehearsal archive was not freshly downloaded')
            initial, summary['stages']['initial'] = render_stage(
                exporter, workspace, destination, 'initial', initial_check)
            require_forecast(initial)
            summary['checks']['fresh_download_and_official_signatures'] = True

            disconnected_check = check_feed(workspace, ROOT, disconnected=True)
            summary['stages']['disconnected'] = {'collector': disconnected_check}
            if disconnected_check.get('network_request_skipped'):
                raise RehearsalUnavailable(disconnected_check['error'])
            if (disconnected_check.get('error_kind') != 'download_error'
                    or 'Injected operational rehearsal disconnection' not in disconnected_check.get('error', '')):
                raise AssertionError('The disconnection case did not exercise the injected network failure')
            failed, summary['stages']['disconnected'] = render_stage(
                exporter, workspace, destination, 'disconnected', disconnected_check)
            if failed.get('health') != 'update_failed' or not failed.get('retained_previous_estimate'):
                raise AssertionError('The failed update did not explicitly retain the previous estimate')
            for field in PRESERVED_FIELDS:
                if failed.get(field) != initial.get(field):
                    raise AssertionError(f'Disconnection changed the previous {field}')
            summary['checks']['outage_preserved_fields'] = list(PRESERVED_FIELDS)

            recovery_check = check_feed(workspace, ROOT)
            summary['stages']['recovered'] = {'collector': recovery_check}
            require_live_success(recovery_check, 'Recovery fetch')
            recovered, summary['stages']['recovered'] = render_stage(
                exporter, workspace, destination, 'recovered', recovery_check)
            require_forecast(recovered)
            if recovered.get('retained_previous_estimate'):
                raise AssertionError('Recovery did not replace the failed-update state')
            unchanged = recovery_check['archive_sha256'] == initial_check['archive_sha256']
            if unchanged and recovered['history'] != initial['history']:
                raise AssertionError('An unchanged recovered archive duplicated forecast history')
            summary['checks']['live_recovery'] = True
            summary['checks']['recovery_archive_unchanged'] = unchanged
            summary['status'] = 'passed'
        except RehearsalUnavailable as error:
            summary.update(status='pending', error=str(error))
        except Exception as error:
            summary.update(status='failed', error=f'{type(error).__name__}: {error}')
        finally:
            try:
                share_cooldown(workspace, ROOT)
                # Preserve the exact frozen model/input files and signed feed
                # cache used here. Nothing is copied into the production cache.
                for name in ('data', 'val2026', 'model-lock.json', 'uv.lock'):
                    source = workspace / name
                    if source.is_dir():
                        shutil.copytree(source, destination / 'evidence' / name)
                    elif source.exists():
                        target = destination / 'evidence' / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                for name in ('outputs/selected-model.json',):
                    source = workspace / name
                    if source.exists():
                        target = destination / 'evidence' / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                # Retain the surrounding collector/notebook code as well as
                # the frozen model so the evidence can be rendered offline.
                code_files = [*sorted((ROOT / 'val2026').glob('*.py')),
                              *sorted((ROOT / 'scripts').glob('*.py')),
                              ROOT / 'election-night.py', ROOT / 'update-report.py',
                              ROOT / 'pyproject.toml']
                for source in code_files:
                    target = destination / 'evidence' / source.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            except Exception as error:
                summary.update(status='failed', finalization_error=f'{type(error).__name__}: {error}')
            summary.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                           runtime_seconds=perf_counter() - started)
            atomic_write(destination / 'summary.json', json_bytes(summary))
            atomic_write(output / 'summary.json', json_bytes(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# %%
if __name__ == '__main__':
    result = main()
    sys.exit(0 if result['status'] == 'passed' else 2 if result['status'] == 'pending' else 1)
