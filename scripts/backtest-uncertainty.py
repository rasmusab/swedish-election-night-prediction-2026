# %%
"""Evaluate approximate Bayesian final-result simulations in the 2018→2022 replay.

Run: uv run scripts/backtest-uncertainty.py --draws 256
This is a diagnostic on one election, not an estimate of future calibration.
Final 2022 results enter scoring only. No uncertainty settings are fitted or
selected using those results. The default ablations are declared below.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/val2026-matplotlib')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from val2026 import ROOT
from val2026.election_data import PARTIES
from val2026.election_forecast import frozen_model
from val2026.forecast_model import SwingRegression
from val2026.replay import FRACTIONS, SCENARIOS, arrival_order, load_backtest, reveal
from val2026.seats import allocate_riksdag

OUT = ROOT / 'outputs'
PREFIX = 'predictive-uncertainty'
HISTORY = ROOT / 'data/seats/history'
NAMED_PARTIES = [p for p in PARTIES if p != 'ÖVR']
BLOCS = {'V+S+MP+C': ['V', 'S', 'MP', 'C'], 'M+L+KD+SD': ['M', 'L', 'KD', 'SD']}
VARIANTS = {
    'full': {},
    'no_reporting_discrepancy': {'reporting_discrepancy': False},
    'no_late_effect': {'late_effect': False, 'late_residuals': False},
    'no_final_revisions': {'final_revisions': False},
    'half_systematic_scale': {'reporting_discrepancy_multiplier': .25, 'late_effect_multiplier': .5,
                              'late_log_volume_sd_floor': .10, 'revision_share_sd': .00025,
                              'revision_log_volume_sd': .0005},
    'double_systematic_scale': {'reporting_discrepancy_multiplier': 1., 'late_effect_multiplier': 2.,
                                'late_log_volume_sd_floor': .40, 'revision_share_sd': .001,
                                'revision_log_volume_sd': .002},
}
ABLATION_SCENARIOS = ['archived_report_time', 'big_cities_late']
ABLATION_FRACTIONS = [.01, .10, .50, 1.0]
LIMITATIONS = [
    'One target election: parties, checkpoints, blocs and reporting scenarios are dependent. Coverage and Brier scores are descriptive, not proof of future calibration.',
    'The point model was previously selected using this 2022 replay, so this is not an untouched test election. Uncertainty defaults are explicit assumptions, not tuned here.',
    'Archived report times are surviving report/correction timestamps, not necessarily first appearances. Other arrival orders are synthetic stress tests.',
    'Late collection pools stay outstanding at 100% ordinary reporting; all forecasts target final votes and seats, including possible ordinary recount revisions.',
    'The eight individually modelled parties determine seat and threshold metrics. Other parties remain in the vote denominator but are pooled and excluded from seats.',
    'A central interval can hide disconnected seat outcomes around 4%. Draw-level seat distributions and threshold probabilities are retained.',
    'Monte Carlo resolution is limited by the draw count. No probability estimate of zero or one establishes impossibility or certainty.',
]


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


# %% Freeze geographic membership and outcomes independently of construction.
def load_history():
    features, preliminary, final = load_backtest()
    membership = pd.read_csv(HISTORY / 'backtest-2022-constituencies.csv', dtype=str)
    if membership.code.duplicated().any() or set(membership.code) != set(features.code):
        raise ValueError('Historical constituency membership must match every feature row exactly')
    aligned = membership.set_index('code').loc[features.code].reset_index()
    for column in ['municipality', 'kind']:
        if not np.array_equal(features[column].to_numpy(), aligned[column].to_numpy()):
            raise ValueError(f'Historical membership differs from the feature {column}')
    features = features.copy()
    features['constituency'] = aligned.constituency.to_numpy()
    fixture = json.loads((HISTORY / '2022.json').read_text())
    if set(features.constituency) != set(fixture['fixed_seats']):
        raise ValueError('Historical fixed seats must cover all feature constituencies')
    final_grouped = aggregate_votes(final, features.constituency)
    if final_grouped != fixture['votes']:
        raise ValueError('Historical final vote totals disagree with the scoring fixture')
    return features, preliminary, final, fixture


def aggregate_votes(predicted, constituencies):
    table = pd.DataFrame(predicted, columns=PARTIES)
    table['constituency'] = np.asarray(constituencies)
    return {code: {p: float(row[p]) for p in PARTIES}
            for code, row in table.groupby('constituency')[PARTIES].sum().iterrows()}


def allocate_draws(vote_draws, constituency_codes, fixed_seats, *, seed):
    """Allocate each complete joint election, preserving the discrete threshold."""
    vote_draws = np.asarray(vote_draws)
    if vote_draws.ndim != 3 or vote_draws.shape[1:] != (len(constituency_codes), len(PARTIES)):
        raise ValueError('Draws require draw × constituency × party dimensions')
    seats = np.zeros((len(vote_draws), len(PARTIES)), dtype=np.int16)
    ties = 0
    for draw_index, votes in enumerate(vote_draws):
        allocation = allocate_riksdag(
            {code: dict(zip(PARTIES, map(float, votes[c])))
             for c, code in enumerate(constituency_codes)}, fixed_seats,
            tie_seed=seed + draw_index)
        seats[draw_index] = [allocation['national_seats'].get(p, 0) for p in PARTIES]
        ties += len(allocation['ties'])
    if not np.all(seats.sum(axis=1) == 349):
        raise AssertionError('Every simulated election must allocate exactly 349 seats')
    return seats, ties


# %% Score supplied simulations; these functions cannot alter their construction.
def interval_metrics(samples, truth, point, entities, metric, *, discrete=False):
    samples = np.asarray(samples)
    truth, point = np.asarray(truth), np.asarray(point)
    if samples.ndim != 2 or samples.shape[1] != len(entities):
        raise ValueError('Samples require one column per scored entity')
    if not np.isfinite(samples).all() or len(samples) < 2:
        raise ValueError('At least two finite draws are required for interval scoring')
    rows = []
    for nominal in [.50, .90]:
        tail = (1 - nominal) / 2
        lower, upper = np.quantile(samples, [tail, 1 - tail], axis=0,
                                   method='inverted_cdf' if discrete else 'linear')
        for i, entity in enumerate(entities):
            rows.append({
                'metric': metric, 'entity': entity, 'nominal': nominal,
                'point': float(point[i]), 'median': float(np.median(samples[:, i])),
                'mean': float(samples[:, i].mean()), 'lower': float(lower[i]),
                'upper': float(upper[i]), 'width': float(upper[i] - lower[i]),
                'actual': float(truth[i]), 'covered': bool(lower[i] <= truth[i] <= upper[i]),
                'distance_outside': float(max(lower[i] - truth[i], truth[i] - upper[i], 0)),
            })
    return rows


def event_metrics(seat_draws, actual_seats, national_shares, actual_shares):
    rows = []
    for bloc, parties in BLOCS.items():
        ids = [PARTIES.index(p) for p in parties]
        event = seat_draws[:, ids].sum(axis=1) >= 175
        actual = sum(actual_seats[p] for p in parties) >= 175
        probability = float(event.mean())
        rows.append({'event': 'majority', 'entity': bloc, 'probability': probability,
                     'actual': bool(actual), 'brier': (probability - actual) ** 2,
                     'monte_carlo_se': float(np.sqrt(probability * (1 - probability) / len(event)))})
    for party in NAMED_PARTIES:
        i = PARTIES.index(party)
        probability = float((national_shares[:, i] >= .04).mean())
        actual = bool(actual_shares[i] >= .04)
        rows.append({'event': 'national_4pct_threshold', 'entity': party,
                     'probability': probability, 'actual': actual,
                     'brier': (probability - actual) ** 2,
                     'monte_carlo_se': float(np.sqrt(probability * (1 - probability) / len(seat_draws)))})
    return rows


def score_case(simulation, observed, predicted, features, fixture, final, *, seed):
    started = perf_counter()
    seat_draws, ties = allocate_draws(simulation.vote_draws, simulation.constituency_codes,
                                    fixture['fixed_seats'], seed=seed)
    allocation_seconds = perf_counter() - started
    national_counts = simulation.vote_draws.sum(axis=1)
    share_draws = national_counts / national_counts.sum(axis=1, keepdims=True)
    point_shares = predicted.sum(axis=0) / predicted.sum()
    final_shares = final.sum(axis=0) / final.sum()
    point_seats = allocate_riksdag(aggregate_votes(predicted, features.constituency),
                                  fixture['fixed_seats'])['national_seats']
    ids = [PARTIES.index(p) for p in NAMED_PARTIES]
    rows = interval_metrics(share_draws * 100, final_shares * 100, point_shares * 100,
                            PARTIES, 'national_share_pct')
    rows += interval_metrics(seat_draws[:, ids], [fixture['national_seats'][p] for p in NAMED_PARTIES],
                             [point_seats[p] for p in NAMED_PARTIES], NAMED_PARTIES,
                             'party_seats', discrete=True)
    bloc_draws = np.column_stack([seat_draws[:, [PARTIES.index(p) for p in ps]].sum(axis=1)
                                 for ps in BLOCS.values()])
    rows += interval_metrics(bloc_draws,
                             [sum(fixture['national_seats'][p] for p in ps) for ps in BLOCS.values()],
                             [sum(point_seats[p] for p in ps) for ps in BLOCS.values()],
                             list(BLOCS), 'bloc_seats', discrete=True)
    events = event_metrics(seat_draws, fixture['national_seats'], share_draws, final_shares)
    diagnostics = {'allocation_seconds': allocation_seconds, 'allocation_ties': ties,
                   'reported_ordinary': int(np.isfinite(observed).all(axis=1).sum()),
                   'max_monte_carlo_probability_se': .5 / np.sqrt(len(seat_draws))}
    return rows, events, diagnostics, {'national_shares': share_draws, 'seats': seat_draws,
                                     'bloc_seats': bloc_draws}


def markdown_table(frame, formatters=None):
    """Avoid an extra table-rendering dependency in the frozen environment."""
    formatters = formatters or {}
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    for _, row in frame.iterrows():
        values = [formatters.get(c, str)(row[c]) for c in frame.columns]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


# %% Report failures by arrival pattern and stage, without claiming calibration.
def render_report(intervals, events, diagnostics, manifest, output_prefix):
    full = intervals[intervals.variant.eq('full')]
    summary = (full.groupby(['scenario', 'fraction', 'metric', 'nominal'], as_index=False)
               .agg(coverage=('covered', 'mean'), mean_width=('width', 'mean'),
                    worst_miss=('distance_outside', 'max')))
    summary.to_csv(OUT / f'{output_prefix}-summary.csv', index=False)
    archived = summary[summary.scenario.eq('archived_report_time') & summary.nominal.eq(.90)]
    archived_pivot = archived.pivot(index='fraction', columns='metric', values=['coverage', 'mean_width'])
    rows = []
    for fraction in sorted(archived.fraction.unique()):
        row = archived_pivot.loc[fraction]
        rows.append({'Reporting': f'{fraction:.0%}',
                     'Share coverage / width': f'{row["coverage", "national_share_pct"]:.0%} / {row["mean_width", "national_share_pct"]:.2f} pp',
                     'Party seats coverage / width': f'{row["coverage", "party_seats"]:.0%} / {row["mean_width", "party_seats"]:.1f}',
                     'Bloc seats coverage / width': f'{row["coverage", "bloc_seats"]:.0%} / {row["mean_width", "bloc_seats"]:.1f}'})
    full_events = events[events.variant.eq('full') & events.event.eq('majority')
                         & events.entity.eq('V+S+MP+C')].copy()
    probability_table = full_events.pivot(index='scenario', columns='fraction', values='probability')
    probability_table.columns = [f'{c:.0%}' for c in probability_table.columns]
    probability_table = probability_table.reset_index()
    failure = full[full.nominal.eq(.90) & ~full.covered & full.metric.ne('national_share_pct')]
    failure_table = failure[['scenario', 'fraction', 'metric', 'entity', 'lower', 'upper', 'actual']].copy()
    failure_table['fraction'] = failure_table.fraction.map(lambda x: f'{x:.0%}')
    ablation = (intervals[intervals.scenario.isin(ABLATION_SCENARIOS)
                         & intervals.fraction.isin(ABLATION_FRACTIONS) & intervals.nominal.eq(.90)]
                .groupby(['variant', 'scenario', 'metric'], as_index=False)
                .agg(coverage=('covered', 'mean'), mean_width=('width', 'mean')))
    ablation.to_csv(OUT / f'{output_prefix}-ablations.csv', index=False)
    runtime = diagnostics[diagnostics.variant.eq('full')]
    report = (
        '# Approximate Bayesian uncertainty: historical diagnostic\n\n'
        'The final 2022 outcome is scored after constructing each forecast from 2018 background '
        'data and the revealed preliminary 2022 districts. Every simulated result is aggregated '
        'to its 29 constituencies and receives its own complete 349-seat allocation. The final '
        'bloc totals were **V+S+MP+C: 173** and **M+L+KD+SD: 176**.\n\n'
        'The defaults were fixed before this evaluation and are recorded verbatim in the config '
        'file. This evaluation does not fit variance scales or choose settings using final 2022 '
        'outcomes. The underlying point forecast was selected earlier on this election.\n\n'
        f'Each checkpoint uses **{manifest["draws_per_checkpoint"]} draws**. The largest possible '
        f'Monte Carlo standard error for a probability is {100 * .5 / np.sqrt(manifest["draws_per_checkpoint"]):.1f} '
        'percentage points; tail quantiles also have simulation noise. Increase the draw count '
        'for the published report. Central seat intervals use draw order statistics, preserving '
        'integer endpoints. Share widths are percentage points.\n\n'
        '## Archived arrival order: nominal 90% intervals\n\n'
        + markdown_table(pd.DataFrame(rows)) + '\n\n'
        'Coverage here is the fraction of parties or blocs whose final outcome falls inside '
        'the interval at that checkpoint. The two blocs are complementary, so their coverage '
        'outcomes are not independent. Seat quantisation can give zero-width intervals that '
        'still cover a final integer result.\n\n'
        f'![Coverage by scenario and stage]({output_prefix}-coverage.png)\n\n'
        '## Majority probabilities: V+S+MP+C reaching 175\n\n'
        'These are model probabilities for the same event at many dependent checkpoints. '
        'That event did not occur in the final 2022 result; the corresponding Brier score '
        'is the squared probability. The second bloc’s scores are equivalent here. They '
        'are retained in the event CSV, not interpreted as election-level calibration.\n\n'
        + markdown_table(probability_table, {c: lambda x: f'{x:.1%}' for c in probability_table if c != 'scenario'})
        + '\n\n'
        f'![Examples of joint bloc outcomes]({output_prefix}-examples.png)\n\n'
        '## Misses of the nominal 90% seat intervals\n\n'
        + (markdown_table(failure_table, {c: lambda x: f'{x:.0f}' for c in ['lower', 'upper', 'actual']})
           if len(failure_table) else 'No party or bloc seat intervals missed in these specific cases.')
        + '\n\n'
        '## Predeclared component ablations\n\n'
        'The archived and delayed-big-city orders are checked at 1%, 10%, 50% and 100%. '
        'Alternatives remove the named extra component or halve/double the systematic-error scales. In particular, removing '
        'the reporting discrepancy does not remove the correlated county and municipality '
        'coefficient uncertainty already in the regression. These tests diagnose assumptions; '
        'they are not a search for the configuration with the best 2022 coverage.\n\n'
        + markdown_table(ablation, {'coverage': lambda x: f'{x:.0%}', 'mean_width': lambda x: f'{x:.2f}'})
        + '\n\n'
        '## Runtime and reproducibility\n\n'
        f'The slowest full checkpoint took **{runtime.checkpoint_seconds.max():.2f} seconds**; '
        f'the median took **{runtime.checkpoint_seconds.median():.2f} seconds**. These timings '
        'include the vote fit, joint simulation, all simulated seat allocations and scoring; '
        'HTML rendering and network download are outside this backtest.\n\n'
        f'Config and source hashes: `{output_prefix}-config.json`. All draw-level national shares, '
        f'party seats and bloc seats are retained in `{output_prefix}-draws/`. Inputs, seeds, '
        'code, component settings, runtime and diagnostics are recorded so that the calculation '
        'can be repeated.\n\n'
        '## Limits\n\n' + '\n'.join('- ' + text for text in LIMITATIONS) + '\n')
    (OUT / f'{output_prefix}-report.md').write_text(report)
    plot_summary(summary, output_prefix)
    plot_examples(output_prefix)


def plot_summary(summary, output_prefix):
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.4), layout='constrained')
    metrics = [('national_share_pct', 'National vote share', 'Percentage points'),
               ('party_seats', 'Party seats', 'Seats'), ('bloc_seats', 'Bloc seats', 'Seats')]
    colors = plt.get_cmap('tab10').colors
    for column, (metric, label, unit) in enumerate(metrics):
        for index, scenario in enumerate(SCENARIOS):
            rows = summary[summary.metric.eq(metric) & summary.scenario.eq(scenario) & summary.nominal.eq(.9)]
            if rows.empty:
                continue
            axes[0, column].plot(rows.fraction * 100, rows.coverage * 100, marker='o',
                                 color=colors[index], label=scenario.replace('_', ' '))
            axes[1, column].plot(rows.fraction * 100, rows.mean_width, marker='o', color=colors[index])
        axes[0, column].axhline(90, color='#222222', linestyle='--', linewidth=1)
        axes[0, column].set(title=label, ylabel='Empirical 90% interval coverage (%)', ylim=(-4, 104))
        axes[1, column].set(ylabel=f'Mean 90% interval width ({unit.lower()})',
                            xlabel='Ordinary districts reporting (%)')
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(fontsize=7, frameon=False, loc='lower right')
    fig.suptitle('2018 → 2022 · one-election uncertainty stress test', fontsize=14)
    fig.savefig(OUT / f'{output_prefix}-coverage.png', dpi=160)
    plt.close(fig)


def plot_examples(output_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout='constrained')
    for ax, scenario in zip(axes, ABLATION_SCENARIOS):
        path = OUT / f'{output_prefix}-draws/{scenario}-1000-full.npz'
        if not path.exists():
            ax.text(.5, .5, '10% checkpoint not requested', ha='center', transform=ax.transAxes)
            continue
        with np.load(path) as saved:
            seats = saved['bloc_seats'][:, 0]
        values, counts = np.unique(seats, return_counts=True)
        ax.bar(values, counts / counts.sum(), color='#85558e', width=.85)
        ax.axvline(175, color='#252525', linestyle='--', linewidth=1.3, label='175-seat majority')
        ax.axvline(173, color='#2a8863', linewidth=1.5, label='Final result: 173')
        ax.set(title=scenario.replace('_', ' ').capitalize(), xlabel='V + S + MP + C seats',
               ylabel='Fraction of simulated elections')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Joint bloc seat outcomes at 10% ordinary reporting', fontsize=13)
    fig.savefig(OUT / f'{output_prefix}-examples.png', dpi=160)
    plt.close(fig)


# %% Run declared scenarios without providing final outcomes to the simulator.
def main(argv=None):
    from val2026.uncertainty import UncertaintyConfig, simulate_constituency_votes
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draws', type=int, default=256)
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument('--fractions', type=float, nargs='+', default=FRACTIONS)
    parser.add_argument('--no-ablations', action='store_true')
    parser.add_argument('--output-prefix', default=PREFIX)
    args = parser.parse_args(argv)
    if args.draws < 20 or any(not 0 < f <= 1 for f in args.fractions):
        parser.error('Use at least 20 draws and fractions in (0, 1]')
    if '/' in args.output_prefix or not args.output_prefix.startswith(PREFIX):
        parser.error('Output prefix must start with predictive-uncertainty and contain no slash')
    started = perf_counter()
    features, preliminary, final, fixture = load_history()
    selected, options, lock = frozen_model(ROOT)
    model = SwingRegression(features, **options)
    default_config = asdict(UncertaintyConfig())
    paths = ['val2026/uncertainty.py', 'val2026/forecast_model.py', 'val2026/replay.py',
             'val2026/seats.py', 'scripts/backtest-uncertainty.py', 'outputs/selected-model.json',
             'data/processed/backtest-2018-2022.csv', 'data/processed/2022-report-times.csv',
             'data/seats/history/2022.json', 'data/seats/history/backtest-2022-constituencies.csv', 'uv.lock']
    manifest = {'model': selected, 'model_options': options, 'model_lock': lock,
                'uncertainty_config': default_config, 'uncertainty_config_sha256': json_hash(default_config),
                'sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
                'draws_per_checkpoint': args.draws, 'scenarios': args.scenarios,
                'fractions': args.fractions, 'arrival_seeds': {'archived_report_time': 0, 'synthetic': 100},
                'simulation_seed_rule': '2026000 + SCENARIOS index * 10000 + round(fraction * 1000)',
                'variants': VARIANTS if not args.no_ablations else {'full': {}},
                'ablation_scenarios': ABLATION_SCENARIOS, 'ablation_fractions': ABLATION_FRACTIONS,
                'bloc_parties': BLOCS, 'intervals': [.50, .90], 'limitations': LIMITATIONS}
    write_json(OUT / f'{args.output_prefix}-config.json', manifest)
    draws_folder = OUT / f'{args.output_prefix}-draws'
    draws_folder.mkdir(parents=True, exist_ok=True)
    all_intervals, all_events, all_diagnostics = [], [], []
    for scenario in args.scenarios:
        arrival_seed = 0 if scenario == 'archived_report_time' else 100
        order = arrival_order(features, scenario, arrival_seed)
        for fraction in args.fractions:
            observed = reveal(preliminary, order, fraction)
            predicted = model.fit_predict(observed)
            variants = VARIANTS if (not args.no_ablations and scenario in ABLATION_SCENARIOS
                                    and fraction in ABLATION_FRACTIONS) else {'full': {}}
            draw_seed = 2026000 + SCENARIOS.index(scenario) * 10000 + round(fraction * 1000)
            for variant, overrides in variants.items():
                checkpoint_started = perf_counter()
                config = {**default_config, **overrides}
                keys = {'scenario': scenario, 'fraction': fraction, 'arrival_seed': arrival_seed,
                        'variant': variant, 'draw_seed': draw_seed,
                        'config_sha256': json_hash(config), 'draws': args.draws}
                simulation = simulate_constituency_votes(model, observed, predicted=predicted,
                                                         draws=args.draws, seed=draw_seed, config=config)
                rows, events, allocation_diagnostics, retained_draws = score_case(
                    simulation, observed, predicted, features, fixture, final, seed=draw_seed)
                all_intervals.extend({**keys, **row} for row in rows)
                all_events.extend({**keys, **row} for row in events)
                diagnostic = {**keys, **simulation.diagnostics, **allocation_diagnostics,
                              'checkpoint_seconds': perf_counter() - checkpoint_started}
                # Preserve structured simulator diagnostics without coercing them into CSV cells.
                all_diagnostics.append(diagnostic)
                case = f'{scenario}-{round(fraction * 10000):04d}-{variant}'
                np.savez_compressed(draws_folder / f'{case}.npz', **retained_draws)
                write_json(draws_folder / f'{case}.json', {'keys': keys, 'config': config,
                                                         'diagnostics': diagnostic})
                interval_frame = pd.DataFrame(all_intervals)
                interval_frame.to_csv(OUT / f'{args.output_prefix}-intervals.csv', index=False)
                pd.DataFrame(all_events).to_csv(OUT / f'{args.output_prefix}-events.csv', index=False)
                write_json(OUT / f'{args.output_prefix}-diagnostics.json', all_diagnostics)
                party_rows = [r for r in rows if r['metric'] == 'party_seats' and r['nominal'] == .90]
                bloc_rows = [r for r in rows if r['metric'] == 'bloc_seats' and r['nominal'] == .90]
                print(f'{scenario} {fraction:.0%} {variant}: '
                      f'90% party-seat coverage {np.mean([r["covered"] for r in party_rows]):.0%}; '
                      f'bloc range {bloc_rows[0]["lower"]:.0f}–{bloc_rows[0]["upper"]:.0f}; '
                      f'{diagnostic["checkpoint_seconds"]:.2f}s', flush=True)
    manifest['runtime_seconds'] = perf_counter() - started
    manifest['cases'] = len(all_diagnostics)
    manifest['completed'] = True
    write_json(OUT / f'{args.output_prefix}-config.json', manifest)
    render_report(pd.DataFrame(all_intervals), pd.DataFrame(all_events),
                  pd.DataFrame(all_diagnostics), manifest, args.output_prefix)
    print(f'Finished {len(all_diagnostics)} cases in {manifest["runtime_seconds"]:.1f}s. '
          f'Report: outputs/{args.output_prefix}-report.md', flush=True)
    return pd.DataFrame(all_intervals), pd.DataFrame(all_events)


# %%
if __name__ == '__main__':
    intervals, events = main()
