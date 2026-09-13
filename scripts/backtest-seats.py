# %%
"""Validate seat allocation, then score the frozen 2018→2022 rolling forecast.

Run: uv run scripts/backtest-seats.py
The 2022 constituency boundaries and fixed seats were known before election
night. Final 2022 votes and seats are used only to validate and score results.
No model selection, fitting changes or uncertainty estimates are performed.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/val2026-matplotlib')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from val2026 import ROOT
from val2026.election_data import PARTIES
from val2026.forecast_model import SwingRegression
from val2026.election_forecast import frozen_model
from val2026.replay import FRACTIONS, SCENARIOS, arrival_order, load_backtest, reveal, simple_forecast

HISTORY = ROOT / 'data/seats/history'
OUT = ROOT / 'outputs'
FIXED_URL = ('https://www.val.se/download/18.4005a7d19dee20a8ea544/'
             '1778074856144/valkretsmandat-riksdag-1988-2026.xlsx')
ARCHIVE_URL = ('https://resultat.val.se/resultatfiler/val2022/s/rd/'
               'Val_20220911_slutlig_00_RD.zip')
SOURCE_2018 = ('https://www.val.se/english/election-results/'
               'elections-to-the-riksdag-and-regional-and-municipal-councils/'
               'election-results-2018')
NAMED_PARTIES = [p for p in PARTIES if p != 'ÖVR']
BLOCS = {'V+S+MP+C': ['V', 'S', 'MP', 'C'], 'M+L+KD+SD': ['M', 'L', 'KD', 'SD']}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def canonical_name(value):
    return re.sub(r'\s*\d\)', '', str(value)).strip().replace('Kopparbergs län/', '')


def official_seats(rows):
    result = {p: 0 for p in PARTIES}
    for row in rows:
        party = row['partiforkortning']
        if party not in result and row['antalMandat']:
            raise ValueError(f'Unmodelled party won historical seats: {party}')
        result[party] = row['antalMandat']
    return result


# %% Derive compact, auditable validation fixtures from the official source data.
def prepare_history():
    HISTORY.mkdir(parents=True, exist_ok=True)
    workbook = HISTORY / 'valkretsmandat-riksdag-1988-2026.xlsx'
    if not workbook.exists():
        with urlopen(FIXED_URL, timeout=45) as response:
            workbook.write_bytes(response.read())
    fixed_table = pd.read_excel(workbook, header=4)
    fixed_table = fixed_table[fixed_table.Valkrets.notna() & fixed_table[2022].notna()].copy()
    fixed_table['name'] = fixed_table.Valkrets.map(canonical_name)
    if len(fixed_table) != 29 or fixed_table['name'].duplicated().any():
        raise ValueError('Expected 29 unique modern constituencies in the fixed-seat workbook')
    fixed_table = fixed_table.set_index('name')

    archive = ROOT / 'data/2022-feed/Val_20220911_slutlig_00_RD.zip'
    with ZipFile(archive) as source:
        result = json.loads(source.read('Val_20220911_slutlig_mandatfordelning_00_RD.json'))
        districts = json.loads(source.read('Val_20220911_slutlig_rostfordelning_00_RD.json'))['valdistrikt']
    area = result['valomrade']
    fixture_2022 = {'year': 2022, 'votes': {}, 'fixed_seats': {}, 'names': {},
                    'national_seats': official_seats(area['mandatfordelning']['partiLista']),
                    'constituencies': {}, 'source': ARCHIVE_URL,
                    'source_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    for constituency in area['valkretsLista']:
        code, name = constituency['kod'], constituency['namnValkrets']
        vote_rows = constituency['rostfordelning']['rosterPaverkaMandat']
        # Retain all valid votes in the threshold denominator. Other registered
        # parties are a pooled residual and cannot be allocated seats together.
        votes = {p: 0 for p in PARTIES}
        for row in vote_rows['partiRoster']:
            party = row['partiforkortning']
            votes[party if party in NAMED_PARTIES else 'ÖVR'] += row['antalRoster']
        votes['ÖVR'] += vote_rows['rosterOvrigaPartier']['antalRoster']
        if sum(votes.values()) != vote_rows['antalRoster']:
            raise ValueError(f'Constituency {code}: final votes do not reconcile')
        seat_rows = constituency['mandatfordelning']['partiLista']
        fixed_count = sum(row['antalFastaMandat'] for row in seat_rows)
        expected_count = int(fixed_table.loc[canonical_name(name), 2022])
        if fixed_count != expected_count:
            raise ValueError(f'Constituency {code}: archival and pre-election fixed seats differ')
        fixture_2022['votes'][code] = votes
        fixture_2022['fixed_seats'][code] = fixed_count
        fixture_2022['names'][code] = name
        fixture_2022['constituencies'][code] = {
            'fixed': {p: next((r['antalFastaMandat'] for r in seat_rows if r['partiforkortning'] == p), 0)
                      for p in PARTIES},
            'adjustment': {p: next((r['antalUtjamningsmandat'] for r in seat_rows if r['partiforkortning'] == p), 0)
                           for p in PARTIES},
            'total': official_seats(seat_rows),
        }
    if sum(fixture_2022['fixed_seats'].values()) != 310 or len(fixture_2022['votes']) != 29:
        raise ValueError('Invalid 2022 seat fixture coverage')
    if sum(sum(v.values()) for v in fixture_2022['votes'].values()) != 6477970:
        raise ValueError('Historical final 2022 valid-vote total changed')

    # Municipality-to-Riksdag-constituency membership is geographic metadata,
    # independent of the final vote counts in the archive used to recover it.
    municipality_membership = {}
    district_membership = {}
    for district in districts:
        municipality, constituency = district['kommunkod'], district['kretskod']
        if municipality in municipality_membership and municipality_membership[municipality] != constituency:
            raise ValueError(f'Municipality {municipality} crosses constituencies; late pools must be split')
        municipality_membership[municipality] = constituency
        district_membership[district['valdistriktskod']] = constituency
    features, _, final = load_backtest()
    membership = features[['code', 'municipality', 'kind']].copy()
    membership['constituency'] = membership.municipality.map(municipality_membership)
    ordinary = membership.kind.eq('ordinary')
    if (membership.constituency.isna().any()
            or not membership.loc[ordinary, 'code'].map(district_membership).equals(membership.loc[ordinary, 'constituency'])):
        raise ValueError('Incomplete or inconsistent historical constituency mapping')
    grouped_final = aggregate_votes(final, membership.constituency)
    if grouped_final != fixture_2022['votes']:
        raise ValueError('Backtest final vote aggregation disagrees with the official constituency totals')

    scb_file = ROOT / 'old/scb-riksdag-results-2018.xlsx'
    scb = pd.read_excel(scb_file)
    grouped_2018 = scb.groupby('Valkretskod')[PARTIES].sum()
    fixture_2018 = {'year': 2018, 'votes': {}, 'fixed_seats': {}, 'names': {},
                    'national_seats': {'M': 70, 'C': 31, 'L': 20, 'KD': 22, 'MP': 16,
                                       'S': 100, 'V': 28, 'SD': 62, 'ÖVR': 0},
                    'national_seats_source': SOURCE_2018,
                    'vote_source': str(scb_file.relative_to(ROOT)),
                    'vote_source_sha256': hashlib.sha256(scb_file.read_bytes()).hexdigest()}
    for raw_code, votes in grouped_2018.iterrows():
        code = f'{int(raw_code):02d}'
        name = canonical_name(scb.loc[scb.Valkretskod.eq(raw_code), 'Valkretsnamn'].iloc[0])
        fixture_2018['votes'][code] = {p: int(votes[p]) for p in PARTIES}
        fixture_2018['fixed_seats'][code] = int(fixed_table.loc[name, 2018])
        fixture_2018['names'][code] = name
    if len(grouped_2018) != 29 or sum(fixture_2018['fixed_seats'].values()) != 310:
        raise ValueError('Incomplete 2018 constituency coverage')
    for fixture in [fixture_2018, fixture_2022]:
        fixture['fixed_seats_source'] = FIXED_URL
        fixture['fixed_seats_source_sha256'] = hashlib.sha256(workbook.read_bytes()).hexdigest()
        write_json(HISTORY / f"{fixture['year']}.json", fixture)
    membership.to_csv(HISTORY / 'backtest-2022-constituencies.csv', index=False)
    return fixture_2018, fixture_2022, membership.constituency


def aggregate_votes(predicted, constituencies):
    if len(predicted) != len(constituencies):
        raise ValueError('One constituency is required per forecast row')
    table = pd.DataFrame(predicted, columns=PARTIES)
    table['constituency'] = np.asarray(constituencies)
    if table.constituency.isna().any() or not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError('Seat allocation requires finite nonnegative votes and complete geography')
    grouped = table.groupby('constituency')[PARTIES].sum()
    return {code: {p: float(row[p]) for p in PARTIES} for code, row in grouped.iterrows()}


# %% Allocation validation is separate from predictive performance.
def validate_allocator(fixtures):
    from val2026.seats import allocate_riksdag
    rows = []
    for fixture in fixtures:
        started = perf_counter()
        allocation = allocate_riksdag(fixture['votes'], fixture['fixed_seats'])
        seconds = perf_counter() - started
        national_mismatches = {p: [allocation['national_seats'].get(p, 0), count]
                               for p, count in fixture['national_seats'].items()
                               if allocation['national_seats'].get(p, 0) != count}
        constituency_mismatches = []
        for code, truth in fixture.get('constituencies', {}).items():
            for kind in ['fixed', 'adjustment', 'total']:
                for party, count in truth[kind].items():
                    estimate = allocation['constituencies'][code][kind].get(party, 0)
                    if estimate != count:
                        constituency_mismatches.append({'constituency': code, 'kind': kind,
                                                       'party': party, 'actual': count, 'allocated': estimate})
        row = {'year': fixture['year'], 'national_seats': allocation['national_seats'],
               'national_exact': not national_mismatches,
               'constituency_exact': not constituency_mismatches if 'constituencies' in fixture else None,
               'national_mismatches': national_mismatches,
               'constituency_mismatches': constituency_mismatches,
               'allocation_seconds': seconds, 'ties': allocation['ties']}
        rows.append(row)
    write_json(OUT / 'seat-history-validation.json', rows)
    if any(not r['national_exact'] or r['constituency_exact'] is False for r in rows):
        raise AssertionError('Historical seat allocation differs from official results; see seat-history-validation.json')
    return rows


def seat_metrics(seats, actual):
    error = np.array([seats.get(p, 0) - actual.get(p, 0) for p in NAMED_PARTIES])
    values = {'party_seat_mae': float(np.abs(error).mean()), 'max_party_seat_error': int(np.abs(error).max()),
              'seats_to_reassign': int(np.abs(error).sum() // 2)}
    for index, (bloc, parties) in enumerate(BLOCS.items(), 1):
        predicted_bloc = sum(seats.get(p, 0) for p in parties)
        actual_bloc = sum(actual.get(p, 0) for p in parties)
        values.update({f'bloc_{index}_seats': predicted_bloc, f'bloc_{index}_actual': actual_bloc,
                       f'bloc_{index}_error': predicted_bloc - actual_bloc,
                       f'bloc_{index}_abs_error': abs(predicted_bloc - actual_bloc),
                       f'bloc_{index}_majority_correct': (predicted_bloc >= 175) == (actual_bloc >= 175)})
    return values


# %% Re-run the frozen point forecast with the existing reporting-order scenarios.
def run_backtest(fixture_2022, constituencies, *, seeds=(100, 101, 102), scenarios=SCENARIOS):
    from val2026.seats import allocate_riksdag
    features, preliminary, _ = load_backtest()
    config_path = OUT / 'selected-model.json'
    selected, options, lock = frozen_model(ROOT)
    model_config = {'models': {selected: options}}
    models = {selected: SwingRegression(features, **options)}
    metrics_rows, party_rows = [], []
    start = perf_counter()
    for scenario in scenarios:
        for seed in ([0] if scenario == 'archived_report_time' else seeds):
            order = arrival_order(features, scenario, seed)
            for fraction in FRACTIONS:
                observed = reveal(preliminary, order, fraction)
                for name in ['raw_reported', 'county_swing', *models]:
                    started = perf_counter()
                    predicted = (models[name].fit_predict(observed) if name in models
                                 else simple_forecast(features, observed, name))
                    vote_seconds = perf_counter() - started
                    started = perf_counter()
                    allocation = allocate_riksdag(aggregate_votes(predicted, constituencies), fixture_2022['fixed_seats'])
                    seat_seconds = perf_counter() - started
                    seats = allocation['national_seats']
                    key = {'scenario': scenario, 'seed': seed, 'fraction': fraction, 'model': name}
                    metrics_rows.append({**key, **seat_metrics(seats, fixture_2022['national_seats']),
                                         'vote_forecast_seconds': vote_seconds, 'seat_allocation_seconds': seat_seconds,
                                         'ties': len(allocation['ties'])})
                    for party in PARTIES:
                        count, actual = seats.get(party, 0), fixture_2022['national_seats'][party]
                        party_rows.append({**key, 'party': party, 'predicted_seats': count, 'actual_seats': actual,
                                           'seat_error': count - actual})
            print(f'Seat replay: {scenario}, seed {seed} complete', flush=True)
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(OUT / 'seat-backtest-metrics.csv', index=False)
    pd.DataFrame(party_rows).to_csv(OUT / 'seat-backtest-parties.csv', index=False)
    write_json(OUT / 'seat-backtest-config.json', {
        'model_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'model_lock': lock,
        'historical_input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [HISTORY / '2022.json', HISTORY / 'backtest-2022-constituencies.csv',
                                             ROOT / 'data/processed/backtest-2018-2022.csv']},
        'models': model_config['models'], 'seeds': list(seeds), 'scenarios': list(scenarios),
        'fractions': FRACTIONS, 'bloc_1': BLOCS['V+S+MP+C'], 'bloc_2': BLOCS['M+L+KD+SD'],
        'party_mae_denominator': 'The eight individually modelled parties; pooled ÖVR excluded from MAE.',
        'raw_reported_method': 'Reported national shares are applied equally across constituencies as a deliberately simple comparator.',
        'limitations': [
            'Only one target election; reserved reporting orders are not independent election outcomes.',
            'Archived timestamps can be later corrections rather than first appearances.',
            'Late collection pools stay outstanding even at 100% of ordinary districts.',
            'Seat point estimates only; majority correctness is retrospective scoring, not a probability.',
        ], 'runtime_seconds': perf_counter() - start,
    })
    render_summary(metrics, next(iter(models)))
    return metrics


def render_summary(metrics, selected):
    archived = metrics[metrics.scenario.eq('archived_report_time')]
    methods = ['raw_reported', 'county_swing', selected]
    labels = {'raw_reported': 'Reported shares', 'county_swing': 'County swing', selected: 'Frozen forecast'}
    colors = {'raw_reported': '#a0a7ad', 'county_swing': '#3486a4', selected: '#c25b35'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    for method in methods:
        rows = archived[archived.model.eq(method)]
        axes[0].plot(rows.fraction * 100, rows.party_seat_mae, marker='o', color=colors[method], label=labels[method])
        axes[1].plot(rows.fraction * 100, rows.bloc_1_seats, marker='o', color=colors[method], label=labels[method])
    axes[0].set(title='Party seat error', ylabel='Mean absolute error across eight parties')
    axes[1].axhline(175, color='#222222', linestyle='--', linewidth=1, label='175-seat majority')
    axes[1].axhline(173, color='#65836b', linestyle=':', linewidth=1.6, label='Final V + S + MP + C: 173')
    axes[1].set(title='V + S + MP + C seats', ylabel='Projected seats')
    for ax in axes:
        ax.set_xlabel('Ordinary districts reporting (%)')
        ax.grid(axis='y', color='#e0e4e8', linewidth=.7)
        ax.spines[['top', 'right']].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('2018 → 2022 seat forecast · archived report order', fontsize=13)
    fig.savefig(OUT / 'seat-backtest-performance.png', dpi=160)
    fig.savefig(OUT / 'seat-backtest-performance.svg')
    plt.close(fig)

    rows = archived[archived.model.eq(selected)]
    table = ['| Ordinary districts | Party seat MAE | Worst party | V+S+MP+C | M+L+KD+SD | Correct majority |',
             '| --- | --- | --- | --- | --- | --- |']
    for row in rows.itertuples():
        table.append(f'| {row.fraction:.0%} | {row.party_seat_mae:.2f} | {row.max_party_seat_error} | '
                     f'{row.bloc_1_seats} | {row.bloc_2_seats} | {"Yes" if row.bloc_1_majority_correct and row.bloc_2_majority_correct else "No"} |')
    aggregate = metrics.groupby(['fraction', 'model'])[['party_seat_mae', 'bloc_1_abs_error', 'bloc_1_majority_correct']].mean()
    (OUT / 'seat-backtest-summary.md').write_text(
        '# Historical seat forecast check\n\n'
        'The allocator exactly reproduces all party national seats in 2018 and 2022, and every '
        '2022 constituency’s fixed, adjustment and total seats. This validates the vote-to-seat '
        'calculation separately from the forecast.\n\n'
        'The frozen point forecast is evaluated against the final 2022 allocation: V+S+MP+C = 173, '
        'M+L+KD+SD = 176. No model parameters were changed for this evaluation.\n\n'
        '## Archived reporting order\n\n' + '\n'.join(table) + '\n\n'
        '![Seat forecast performance](seat-backtest-performance.png)\n\n'
        '## All reporting scenarios\n\n````\n' + aggregate.round(3).to_string() + '\n````\n\n'
        '“Correct majority” is a historical yes/no score, not a confidence or winner probability. '
        'The archived order contains surviving report/correction timestamps, not a recording of '
        'every first appearance. Other scenarios are synthetic. All scenarios reuse the same '
        '2022 election; they are not independent election-level validation. At 100%, late '
        'collection votes are still estimated. Party MAE covers the eight named parties.\n\n'
        f'Sources: [official 2018 seats]({SOURCE_2018}), [2022 final archive]({ARCHIVE_URL}), '
        f'[official historical fixed seats]({FIXED_URL}).\n')
    print('\nFrozen forecast, archived order:')
    print(rows[['fraction', 'party_seat_mae', 'max_party_seat_error', 'bloc_1_seats', 'bloc_2_seats']].to_string(index=False))


# %%
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    fixture_2018, fixture_2022, constituencies = prepare_history()
    OUT.mkdir(exist_ok=True)
    validation = validate_allocator([fixture_2018, fixture_2022])
    print('Historical allocation validation:', json.dumps(validation, ensure_ascii=False))
    if not args.validate_only:
        return run_backtest(fixture_2022, constituencies)
    return validation


if __name__ == '__main__':
    result = main()
