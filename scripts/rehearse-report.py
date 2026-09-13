# %%
"""Exercise the entire update workflow using a synthetic sequence of rehearsal returns.

uv run scripts/rehearse-report.py
The official rehearsal is artificial and fully counted. This script hides
records, then reveals them, revises a return and injects failures. It uses an
isolated temporary workspace, a simulated transport/signature verifier, and
real parsing, alignment, frozen regression, state handling and HTML execution.
Real signature checks are tested separately and by scripts/collect-results.py.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from time import perf_counter

from val2026.election_feed import FetchError, atomic_write, json_bytes
from val2026.election_forecast import ROOT
from scripts.rehearsal_scenarios import Scenario, partial

spec = importlib.util.spec_from_file_location('update_report', ROOT / 'update-report.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def main():
    started = perf_counter()
    rows = []
    destination = ROOT / 'outputs/rehearsal-drill'
    with tempfile.TemporaryDirectory(prefix='val2026-drill-') as folder:
        scenario = Scenario(Path(folder))
        cases = [("waiting",0,0), ("one_district",1,0), ("first_estimate",2,0),
                 ("early",100,0), ("partial",1200,0), ("partial_late",1200,1)]
        for step, (name,n,late) in enumerate(cases):
            state = scenario.run(partial(scenario.source,n,late,step=step))
            assert state['health'] == 'ok', state.get('error')
            assert state['ordinary_districts_observed'] == n
            assert state['estimate_state'] == ('waiting' if n < 2 else 'early' if n < 300 else 'forecast')
            if n >= 2:
                assert sum(p['forecast_seats'] for p in state['parties']) == 349
                assert sum(b['seats'] for b in state['blocs']) == 349
                assert all(sum(draw) == 349 for draw in state['uncertainty']['seat_draws'])
            rows.append({'case':name,'health':state['health'],'ordinary_reported':n,
                         'fit_seconds':state['fit_predict_seconds']})
            if name in ('waiting','early','partial'):
                exporter.render(scenario.root,'rehearsal',offline=True,output=destination/name/'index.html')
        data = partial(scenario.source,1200,1,step=6)
        record = next(d for d in data['valdistrikt'] if d['rapporteringsTid'])
        valid = record['rostfordelning']['rosterPaverkaMandat']
        valid['partiRoster'][0]['antalRoster'] -= 1
        valid['antalRoster'] -= 1
        corrected = scenario.run(data)
        assert corrected['raw_valid_votes'] == state['raw_valid_votes'] - 1
        rows.append({'case':'downward_correction','health':corrected['health']})
        count = len(corrected['history'])
        unchanged = scenario.run()
        assert len(unchanged['history']) == count
        rows.append({'case':'unchanged','history_not_duplicated':True})
        scenario.client.error = FetchError('Injected rehearsal outage',status=503)
        failed = scenario.run()
        assert failed['parties'] == unchanged['parties'] and failed['health'] == 'update_failed'
        assert failed['seat_allocation'] == unchanged['seat_allocation']
        assert failed['uncertainty'] == unchanged['uncertainty']
        exporter.render(scenario.root,'rehearsal',offline=True,output=destination/'failed-update/index.html')
        rows.append({'case':'outage','last_estimate_retained':True})
        rollback = scenario.run(partial(scenario.source,1200,step=1))
        assert rollback['latest_check']['collection_state'] == 'source_rollback'
        rows.append({'case':'older_snapshot','rejected':True})
        full = scenario.run(partial(scenario.source,100000,100000,step=10))
        assert full['raw_valid_votes'] == full['forecast_valid_votes']
        assert all(abs(p['raw_reported_share_pct']-p['forecast_share_pct']) < 1e-10 for p in full['parties'])
        assert sum(p['forecast_seats'] for p in full['parties']) == 349
        exporter.render(scenario.root,'rehearsal',offline=True,output=destination/'complete/index.html')
        rows.append({'case':'complete','reported_votes_preserved':True,
                     'ordinary_reported':full['ordinary_districts_observed'], 'late_pools_complete':full['complete_late_municipalities']})
    summary = {'status':'passed','runtime_seconds':perf_counter()-started,'cases':rows,
               'scope':'Synthetic sequence of official artificial rehearsal counts. No evidence of 2026 forecast accuracy.'}
    atomic_write(destination/'summary.json',json_bytes(summary))
    print(json.dumps(summary,indent=2))
    return summary


# %%
if __name__ == '__main__':
    results = main()
