# %%
"""Render an isolated, explicitly fictional partial-count report for inspection.

uv run scripts/render-mock.py
Uses artificial official rehearsal votes and ten simulated reporting updates.
Does not change production/rehearsal caches or publish anything remotely.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from val2026 import ROOT
from val2026.election_forecast import load_inputs
from val2026.forecast_uncertainty import DEFAULT_DRAWS
from val2026.election_feed import atomic_write, json_bytes
from val2026.election_report import plot_bloc_seats, plot_bloc_uncertainty, plot_history
from scripts.rehearsal_scenarios import Scenario, partial


# %%
def main():
    destination = ROOT / 'outputs/mock-report'
    destination.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('mock_exporter', ROOT / 'update-report.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    with tempfile.TemporaryDirectory(prefix='val2026-mock-') as temporary:
        scenario = Scenario(Path(temporary), draws=DEFAULT_DRAWS)
        features, _ = load_inputs(scenario.root)
        ordinary = features[features.kind.eq('ordinary')].copy()
        rng = np.random.default_rng(20260913)
        county_offsets = dict(zip(sorted(ordinary.county.unique()), rng.normal(0, .65, 21)))
        ordinary['arrival'] = (rng.normal(size=len(ordinary)) + .45 * np.log(ordinary.electorate / 1000)
                               + ordinary.county.map(county_offsets))
        order = dict(zip(ordinary.sort_values('arrival').code, range(len(ordinary))))
        scenario.source['valdistrikt'].sort(key=lambda d: order.get(d['valdistriktskod'], 100000))
        start = datetime.now(timezone.utc) - timedelta(minutes=90, seconds=30)
        for index, count in enumerate([60, 120, 250, 400, 650, 900, 1200, 1600, 1900, 2200]):
            stamp = start + timedelta(minutes=10 * index)
            data = partial(scenario.source, count, step=index)
            data['senasteUppdateringstid'] = stamp.isoformat()
            for district in data['valdistrikt']:
                if district['rapporteringsTid']:
                    district['rapporteringsTid'] = stamp.isoformat()
            scenario.now = stamp + timedelta(seconds=10)
            state = scenario.run(data)
            if state['health'] != 'ok':
                raise ValueError(f'Mock update failed: {state.get("error", state["health"])}')
        receipt = exporter.render(scenario.root, 'rehearsal', offline=True,
                                  output=destination / 'index.html')
        state = json.loads((scenario.root / 'outputs/latest-rehearsal/report-state.json').read_text())
        html = (destination / 'index.html').read_text()
        html = html.replace('REHEARSAL · ARTIFICIAL RESULTS', 'ONE-OFF MOCK · SIMULATED DATA')
        html = html.replace('These are artificial rehearsal votes. They say nothing about the likely 2026 result.',
                            'Fictional preview using artificial rehearsal votes and simulated timestamps. '
                            '2,200 of 6,312 districts are shown as reporting across ten updates. '
                            'These seat totals say nothing about the likely 2026 result.')
        html = html.replace('Cached data · no live check · Offline run', 'Static mock · no live data')
        assert len(re.findall(r'<img\b[^>]*src="data:image/png;base64,', html)) == 5
        assert not re.search(r'<div[^>]*class="[^"]*jp-InputArea', html)
        atomic_write(destination / 'index.html', html.encode())
        receipt.update(mock=True, html_sha256=hashlib.sha256(html.encode()).hexdigest())
        atomic_write(destination / 'render-receipt.json', json_bytes(receipt))
        atomic_write(destination / 'mock-data.json', json_bytes(state))
        for name, plot in [('bloc-seats', plot_bloc_seats), ('bloc-uncertainty', plot_bloc_uncertainty), ('seat-history', plot_history)]:
            figure = plot(state)
            figure.savefig(destination / f'{name}.png', dpi=160)
            plt.close(figure)
        assert sum(b['seats'] for b in state['blocs']) == 349
    print(f'Mock report: {destination / "index.html"}')
    print('Fictional bloc seats:', {b['name']: b['seats'] for b in state['blocs']})
    return state


# %%
if __name__ == '__main__':
    mock = main()
