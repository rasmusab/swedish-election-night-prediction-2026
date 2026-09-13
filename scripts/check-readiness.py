# %%
"""Check local election readiness and the feed, without making a forecast.

uv run scripts/check-readiness.py
uv run scripts/check-readiness.py --environment rehearsal
uv run scripts/check-readiness.py --offline
Exit codes: 0 checks passed (possibly warnings), 1 failure, 2 pending external check.
"""
import argparse
from val2026 import ROOT
from val2026.election_feed import atomic_write, json_bytes
from val2026.locking import update_lock
from val2026.readiness import run, markdown


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=['production', 'rehearsal'], default='production')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args(argv)
    with update_lock(ROOT):
        report = run(ROOT, environment=args.environment, offline=args.offline)
        # Show diagnostic results even when the output directory cannot be written.
        print(markdown(report), flush=True)
        folder = ROOT / 'outputs/readiness' / args.environment
        try:
            atomic_write(folder / 'latest.json', json_bytes(report))
            atomic_write(folder / 'latest.md', markdown(report).encode())
        except OSError as error:
            print(f'Could not save readiness report: {error}', flush=True)
            return 1
    return report['exit_code']


# %%
if __name__ == '__main__':
    raise SystemExit(main())
