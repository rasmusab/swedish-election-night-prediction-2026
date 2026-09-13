# %%
"""Collect verified Riksdag snapshots; one rehearsal check by default.

uv run scripts/collect-results.py
uv run scripts/collect-results.py --environment production --once
uv run scripts/collect-results.py --environment production --max-checks 60 --interval 60
uv run scripts/collect-results.py --environment production --count final --once

No background process or scheduled task is created. The minimum 30-second
interval is our precaution, not a documented official polling limit. Final-count
files may still be provisional until the authority formally confirms them.
Full district metadata and counts remain in each snapshot's rostfordelning.json.
"""
import argparse
from pathlib import Path
import time

from val2026.election_feed import Collector

from val2026 import ROOT


# %% Bounded command-line collection; defaults also work when run as cells.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=["rehearsal", "production"], default="rehearsal")
    parser.add_argument("--count", choices=["preliminary", "final"], default="preliminary")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="One check (default)")
    mode.add_argument("--max-checks", type=int, default=1, help="Finite number of checks")
    parser.add_argument("--interval", type=float, default=60, help="Seconds between checks (minimum 30)")
    parser.add_argument("--stale-minutes", type=float, default=15, help="Source age flag threshold")
    args = parser.parse_args(argv)
    if args.max_checks < 1 or args.max_checks > 1440:
        parser.error("--max-checks must be between 1 and 1440")
    if args.interval < 30 or args.interval > 3600:
        parser.error("--interval must be between 30 and 3600 seconds")
    if args.stale_minutes <= 0:
        parser.error("--stale-minutes must be positive")
    collector = Collector(ROOT, environment=args.environment, count=args.count,
                          stale_minutes=args.stale_minutes)
    errors = 0
    for check_number in range(args.max_checks):
        receipt = collector.check()
        print(f"[{check_number + 1}/{args.max_checks}] {args.environment} {args.count}: {receipt['status']}", flush=True)
        if receipt["status"] == "ok":
            errors = 0
            print("TEST DATA" if receipt["test"] else "Production data", flush=True)
            print(f"Archive {'changed' if receipt['archive_changed'] else 'unchanged'}; "
                  f"official signatures verified; {receipt['district_records']:,} districts", flush=True)
            print(f"Source updated: {receipt['source_updated_at']} ({receipt['source_freshness']})", flush=True)
            print(f"Counted: {receipt['districts_counted']} / {receipt['districts_expected']}", flush=True)
        else:
            errors += 1
            print(receipt["error"], flush=True)
        print(f"Receipt: {collector.cache / 'latest-check.json'}", flush=True)
        if check_number + 1 < args.max_checks:
            delay = max(args.interval, receipt.get("retry_after_seconds") or 0,
                        min(600, 30 * 2 ** min(errors, 5)) if errors else 0)
            if delay > 3600:
                print(f"Server requests a {delay:.0f}-second pause; stopping this bounded run. Retry later.", flush=True)
                return 1
            print(f"Next check in {delay:.0f} seconds.", flush=True)
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline:
                time.sleep(min(30, deadline - time.monotonic()))
    return int(receipt["status"] != "ok")


# %% Run the collector explicitly; importing it does not download data.
if __name__ == "__main__":
    raise SystemExit(main())
