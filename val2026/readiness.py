"""Election-day preflight. Collect and validate; never fit or publish a forecast."""
from datetime import datetime, timezone
from importlib import import_module, metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
from zoneinfo import ZoneInfo

from val2026.election_feed import (Collector, HttpClient, FetchError, BASE_URLS, CERTIFICATE_URL,
                                  atomic_write, json_bytes, record_retry_deadline)


def feed_check(receipt, environment, now):
    """A reachable but unpublished index is a pending gate, not a passed test."""
    if receipt.get('status') == 'ok':
        return 'pass', 'Index, archive checksum, signed JSON and election metadata verified.'
    if receipt.get('network_request_skipped') or receipt.get('http_status') == 429:
        return 'pending', 'Server-requested cooldown; retry after ' + str(receipt.get('retry_not_before_utc'))
    if receipt.get('http_status') == 404:
        if receipt.get('failed_url') != BASE_URLS[environment] + 'index.md5':
            return 'fail', 'A required feed resource is missing (HTTP 404): ' + str(receipt.get('failed_url'))
        before = now < datetime(2026, 9, 13, 20, tzinfo=ZoneInfo('Europe/Stockholm'))
        if environment == 'production':
            return ('pending' if before else 'fail'), 'Host reachable; production index is not available (HTTP 404).'
        return 'warn', 'Rehearsal index unavailable (HTTP 404); cached rehearsal files can still be used.'
    return 'fail', receipt.get('error', 'Feed check failed.')


def runtime_check(root):
    if sys.version_info < (3, 14):
        raise ValueError('Python 3.14 or newer is required; use the project uv environment.')
    locked = {p['name']: p['version'] for p in tomllib.loads((root / 'uv.lock').read_text())['package']}
    names = {'numpy': 'numpy', 'scipy': 'scipy', 'pandas': 'pandas', 'matplotlib': 'matplotlib',
             'openpyxl': 'openpyxl', 'jupytext': 'jupytext', 'nbformat': 'nbformat',
             'nbclient': 'nbclient', 'nbconvert': 'nbconvert', 'ipykernel': 'ipykernel',
             'jupyter-client': 'jupyter_client'}
    versions = {}
    for package, module in names.items():
        import_module(module)
        versions[package] = metadata.version(package)
        if versions[package] != locked.get(package):
            raise ValueError(f'{package} differs from uv.lock; run uv sync --locked before launch.')
    if not shutil.which('uv'):
        raise ValueError('uv is not available in this terminal.')
    return {'python': sys.version.split()[0], 'executable': sys.executable, 'packages': versions}


def inputs_check(root):
    from val2026.election_forecast import load_inputs, frozen_model, read_hashed
    features, manifest = load_inputs(root)
    model, _, lock = frozen_model(root)
    folder = Path(manifest['baseline_path']).parent
    for source in manifest['sources']:
        read_hashed(root, folder / source['filename'], source['sha256'])
    return {'snapshot_id': manifest['snapshot_id'], 'prepared_at_utc': manifest['prepared_at_utc'],
            'ordinary_districts': int(features.kind.eq('ordinary').sum()),
            'constituencies': int(features.constituency.nunique()), 'model': model,
            'model_lock_id': lock['lock_id']}


def storage_check(root):
    free = shutil.disk_usage(root).free
    if free < 100_000_000:
        raise ValueError('Less than 100 MB free; make room before collecting result archives.')
    for name in ('data/collector', 'outputs'):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.preflight-', dir=folder) as temporary:
            path = Path(temporary) / 'atomic-write'
            atomic_write(path, b'local read/write check')
            if path.read_bytes() != b'local read/write check':
                raise ValueError(f'Local write/read failed under {folder}')
    return {'free_gb': round(free / 1e9, 2)}


def openssl_check():
    executable = shutil.which('openssl')
    if not executable:
        raise ValueError('OpenSSL is required for result signature verification.')
    result = subprocess.run([executable, 'version'], capture_output=True, check=True, timeout=10)
    return {'version': result.stdout.decode().strip(), 'executable': executable}


def certificate_check(client):
    certificate = client.get(CERTIFICATE_URL, 100_000)
    with tempfile.NamedTemporaryFile(suffix='.pem') as handle:
        handle.write(certificate)
        handle.flush()
        result = subprocess.run(['openssl', 'x509', '-in', handle.name, '-noout',
                                 '-checkend', '0', '-enddate', '-fingerprint', '-sha256'],
                                capture_output=True, check=True, timeout=10)
    return {'url': CERTIFICATE_URL, 'certificate': result.stdout.decode().strip()}


def snapshot_check(root, environment):
    from val2026.election_forecast import load_inputs, load_snapshot, resolve_roster, align_observed
    features, _ = load_inputs(root)
    receipt, source = load_snapshot(root, environment)
    roster, complete, provenance = resolve_roster(root, environment, features, source, receipt)
    _, _, raw, alignment = align_observed(features, source['valdistrikt'], roster,
                                          late_roster_complete=complete)
    if source['antalValdistriktRaknade'] != alignment['source_records_reported']:
        raise ValueError('Reported district count does not reconcile with complete party counts.')
    return {'roster_complete': complete, 'roster_status': provenance['status'],
            'aggregate_reconciliation': receipt['aggregate_reconciliation'],
            'reported_districts': alignment['source_records_reported'],
            'reported_valid_votes': int(raw.sum()), 'archive_sha256': receipt['archive_sha256']}


def run(root, *, environment='production', offline=False, client=None, now=None):
    now = now or datetime.now(timezone.utc)
    client = client or HttpClient(timeout=20, max_attempts=2, max_retry_wait=10)
    checks = []

    def add(name, status, detail, evidence=None):
        checks.append({'name': name, 'status': status, 'detail': detail, 'evidence': evidence})

    def attempt(name, function, detail):
        try:
            evidence = function()
            add(name, 'pass', detail, evidence)
            return evidence
        except Exception as error:
            # A diagnostic should report independent failures in one run.
            add(name, 'fail', f'{type(error).__name__}: {error}')
            return None

    attempt('Runtime', lambda: runtime_check(root), 'Required libraries import and match the locked versions.')
    inputs = attempt('Frozen inputs and model', lambda: inputs_check(root), 'All input and model hashes verified; source workbooks readable locally.')
    attempt('OpenSSL', openssl_check, 'Signature verification executable is available.')
    storage = attempt('Local storage', lambda: storage_check(root), 'Archive and report directories support atomic writes.')
    if storage and storage['free_gb'] < 1:
        add('Disk space', 'warn', 'Less than 1 GB free; allow space for archived updates.')
    receipt = None
    if offline:
        add('Live connection', 'pending', 'Offline check: certificate and feed access were not tested.')
    else:
        try:
            receipt = Collector(root, environment=environment, client=client).check()
            status, detail = feed_check(receipt, environment, now)
            add('Live feed', status, detail, receipt)
        except Exception as error:
            add('Live feed', 'fail', f'{type(error).__name__}: {error}')
        if receipt and (receipt.get('network_request_skipped') or receipt.get('retry_not_before_utc')):
            add('Public certificate access', 'pending', 'Deferred with the server-requested cooldown.')
        else:
            try:
                evidence = certificate_check(client)
                add('Public certificate access', 'pass', 'Official HTTPS certificate is accessible and unexpired.', evidence)
            except FetchError as error:
                deadline = record_retry_deadline(root, error)
                add('Public certificate access', 'pending' if deadline else 'fail', str(error),
                    {'retry_not_before_utc': deadline, 'failed_url': error.url})
            except Exception as error:
                add('Public certificate access', 'fail', f'{type(error).__name__}: {error}')
    cache = root / 'data/collector' / environment / 'preliminary'
    if inputs and (cache / 'latest-success.json').exists():
        snapshot = attempt('Snapshot alignment', lambda: snapshot_check(root, environment),
                           'Signed districts match frozen geography and official national, constituency and municipality totals.')
        if snapshot and not snapshot['roster_complete']:
            add('Complete collection roster', 'pending', 'Waiting for a full official roster; late pools remain incomplete.')
        if receipt and receipt.get('status') == 'ok' and receipt.get('source_freshness') != 'recent':
            add('Source age', 'warn', 'Source results are old or undated; successful access does not mean new votes.',
                {'source_updated_at': receipt.get('source_updated_at')})
    else:
        add('Snapshot alignment', 'pending', 'No verified snapshot can yet be checked against the frozen geography.')
    states = {c['status'] for c in checks}
    status = 'needs_attention' if 'fail' in states else 'pending' if 'pending' in states else 'checks_passed'
    return {'checked_at_utc': now.isoformat(), 'environment': environment, 'offline': offline,
            'status': status, 'exit_code': 1 if 'fail' in states else 2 if 'pending' in states else 0,
            'checks': checks}


def markdown(report):
    lines = [f"# Election readiness — {report['environment']}", '',
             f"Checked: {report['checked_at_utc']} · Status: {report['status']}", '']
    lines += [f"- **{item['status'].upper()} — {item['name']}:** {item['detail']}" for item in report['checks']]
    lines += ['', 'This check does not fit a model or publish a forecast. A passed check is not a promise of election-night availability.', '']
    return '\n'.join(lines)
