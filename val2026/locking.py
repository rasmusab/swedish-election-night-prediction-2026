"""Local update lock; kept independent of optional numerical libraries."""
from contextlib import contextmanager
import fcntl
from val2026 import ROOT


@contextmanager
def update_lock(root=ROOT):
    path = root / 'outputs/.report-update.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another report update is still running; this run made no changes.') from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
