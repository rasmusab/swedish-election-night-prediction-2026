# %%
"""One local command: collect → execute #%% notebook → atomically replace HTML.

uv run update-report.py --environment production
uv run update-report.py --environment rehearsal
Add --offline to render already cached data without claiming a new fetch.
No background polling is started. Run again whenever an update is wanted.
Default production runs also update root index.html for GitHub Pages.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from time import perf_counter

import jupytext
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client.kernelspec import KernelSpecManager
from traitlets.config import Config
from val2026.election_feed import atomic_write, json_bytes
from val2026.election_report import ROOT, update_lock


def render(root, environment, *, offline=False, output=None, draws=1000):
    started = perf_counter()
    destination = output or root / 'outputs' / f'report-{environment}' / 'index.html'
    with update_lock(root), tempfile.TemporaryDirectory(prefix='val2026-notebook-') as temporary:
        temporary = Path(temporary)
        # Install a temporary kernel spec pointing at this exact uv interpreter;
        # do not modify the user's global Jupyter kernels.
        kernel = temporary / 'kernels' / 'val2026'
        kernel.mkdir(parents=True)
        (kernel / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'val2026 uv', 'language': 'python'}))
        notebook = jupytext.read(ROOT / 'election-night.py')
        notebook.metadata.kernelspec = {'name': 'val2026', 'display_name': 'val2026 uv', 'language': 'python'}
        notebook.metadata.title = 'Swedish election 2026 — election-night estimate'
        client = NotebookClient(notebook, timeout=850, kernel_name='val2026',
            resources={'metadata': {'path': str(ROOT)}}, allow_errors=False)
        client.create_kernel_manager()
        client.km.transport = 'ipc'
        client.km.kernel_spec_manager = KernelSpecManager(kernel_dirs=[str(temporary / 'kernels')])
        env = dict(os.environ, VAL2026_ENVIRONMENT=environment, VAL2026_PROJECT_ROOT=str(root),
                   VAL2026_OFFLINE='1' if offline else '0', VAL2026_WRAPPER_LOCK='1',
                   VAL2026_UNCERTAINTY_DRAWS=str(draws),
                   MPLBACKEND='Agg', VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        client.execute(env=env)
        config = Config()
        config.HTMLExporter.exclude_input = True
        config.HTMLExporter.exclude_input_prompt = True
        config.HTMLExporter.exclude_output_prompt = True
        config.HTMLExporter.embed_images = True
        exporter = HTMLExporter(config=config, template_name='lab')
        export_notebook = deepcopy(notebook)
        export_notebook.cells = [c for c in export_notebook.cells if c.cell_type != 'markdown']
        html, resources = exporter.from_notebook_node(export_notebook)
        # All outputs are static HTML and embedded PNGs. No external scripts are
        # needed: remove nbconvert's optional math/widget loaders for offline use.
        html = re.sub(r'<script\b[^>]*>.*?</script>', '', html, flags=re.S | re.I)
        if re.search(r'<div[^>]*class="[^"]*jp-InputArea', html):
            raise ValueError('HTML export unexpectedly contains notebook code')
        if '<html' not in html or '</html>' not in html or 'Election-night estimate' not in html:
            raise ValueError('HTML report is incomplete')
        state = json.loads((root / 'outputs' / f'latest-{environment}' / 'report-state.json').read_bytes())
        if state['test'] is not (environment == 'rehearsal'):
            raise ValueError('Rendered report has wrong environment')
        # The public artifact consists of one self-contained file, so replacement
        # is atomic. Failed execution/export leaves the prior HTML untouched.
        atomic_write(destination, html.encode())
        atomic_write(destination.parent / 'executed.ipynb', nbformat.writes(notebook).encode())
        receipt = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                   'environment': environment, 'health': state['health'], 'output': str(destination),
                   'simulation_draws': state.get('uncertainty', {}).get('draws'),
                   'html_sha256': hashlib.sha256(html.encode()).hexdigest(),
                   'runtime_seconds': perf_counter() - started}
        if environment == 'production' and output is None:
            receipt['public_output'] = str(root / 'index.html')
        atomic_write(destination.parent / 'render-receipt.json', json_bytes(receipt))
        # Publish only the default production render to the repository homepage.
        # Keep supporting artifacts local; rehearsal/custom renders are isolated.
        if 'public_output' in receipt:
            atomic_write(root / 'index.html', html.encode())
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=['production','rehearsal'], default='production')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--draws', type=int, default=1000, help='Joint uncertainty simulations (default: 1000)')
    parser.add_argument('--output', type=Path, help='Custom destination for index.html')
    args = parser.parse_args(argv)
    receipt = render(ROOT, args.environment, offline=args.offline,
                     output=args.output.resolve() if args.output else None, draws=args.draws)
    print(f"Report: {receipt.get('public_output', receipt['output'])}\nState: {receipt['health']}\nTotal time: {receipt['runtime_seconds']:.1f}s")
    return receipt


# %%
if __name__ == '__main__':
    main()
