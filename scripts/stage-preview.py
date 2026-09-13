# %%
"""Stage the tested HTML reports for the existing private Sites preview.

This copies public HTML only. It does not upload or deploy.
Run after update-report.py and scripts/rehearse-report.py, then republish the existing Site.
"""
from pathlib import Path
from val2026.election_feed import atomic_write

from val2026 import ROOT
PAGES = {'index.html': 'outputs/report-production/index.html',
         'rehearsal.html': 'outputs/rehearsal-drill/partial/index.html',
         'failed-update.html': 'outputs/rehearsal-drill/failed-update/index.html'}
NAV = '<nav style="padding:16px 24px;background:#182536;color:white;font:16px system-ui"><a style="color:white" href="/">Production report</a> · <a style="color:white" href="/rehearsal.html">Rehearsal example</a> · <a style="color:white" href="/failed-update.html">Failed-update example</a></nav>'

# %%
if __name__ == '__main__':
    for destination, source in PAGES.items():
        html = (ROOT / source).read_text()
        if 'REHEARSAL · ARTIFICIAL RESULTS' not in html and destination != 'index.html':
            raise ValueError('Rehearsal example lacks a visible test-data label')
        # nbconvert may add attributes to body, so insert after its closing >.
        start = html.index('>', html.index('<body')) + 1
        html = html[:start] + NAV + html[start:]
        atomic_write(ROOT / 'site/dist' / destination, html.encode())
    print('Staged production and two explicitly labelled rehearsal HTML reports.')
