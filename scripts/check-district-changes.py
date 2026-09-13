# %%
"""Diagnose changed-district fallback in the frozen 2018→2022 replay.

Run: uv run scripts/check-district-changes.py --draws 256
Final outcomes enter scoring only. This script does not select or change model
settings. Outputs: outputs/district-changes/{report.md,*.csv,*.png,config.json}.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/val2026-matplotlib')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
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
from val2026.election_forecast import frozen_model, load_inputs
from val2026.forecast_model import SwingRegression
from val2026.replay import FRACTIONS, SCENARIOS, arrival_order, reveal
from val2026.seats import allocate_riksdag
from val2026.uncertainty import UncertaintyConfig, simulate_constituency_votes

OUT = ROOT / 'outputs/district-changes'
NAMED = [p for p in PARTIES if p != 'ÖVR']
NAMED_IDS = [PARTIES.index(p) for p in NAMED]
# Declared before observing errors: group by known ordinary electorate, not votes.
EXPOSURE_CUTOFF = 1 / 3
GROUPS = {'district_mapping': 'Comparable', 'municipal_fallback': 'Municipal fallback'}
EXPOSURE_LABELS = {'low': 'At most one-third fallback', 'high': 'Over one-third fallback'}


def history_helpers():
    path = ROOT / 'scripts/backtest-uncertainty.py'
    spec = importlib.util.spec_from_file_location('district_change_history', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def baseline_groups(features):
    allowed = {'district_mapping', 'municipality_residual', 'municipality_average', 'municipality_late'}
    if not features.baseline_source.isin(allowed).all():
        raise ValueError('Unexpected baseline source')
    return features.baseline_source.replace({'municipality_residual': 'municipal_fallback',
                                             'municipality_average': 'municipal_fallback'})


def exposure_table(features):
    """Only pre-election features determine fallback exposure and grouping."""
    ordinary = features.loc[features.kind.eq('ordinary')].copy()
    groups = baseline_groups(ordinary)
    if not groups.isin(GROUPS).all():
        raise ValueError('Unexpected ordinary baseline source')
    ordinary['fallback_electorate'] = ordinary.electorate.where(
        groups.eq('municipal_fallback'), 0)
    ordinary['fallback_districts'] = groups.eq('municipal_fallback').astype(int)
    result = ordinary.groupby('constituency').agg(
        electorate=('electorate', 'sum'), districts=('code', 'size'),
        fallback_electorate=('fallback_electorate', 'sum'),
        fallback_districts=('fallback_districts', 'sum')).reset_index()
    result['fallback_electorate_fraction'] = result.fallback_electorate / result.electorate
    result['exposure_group'] = np.where(result.fallback_electorate_fraction > EXPOSURE_CUTOFF,
                                        'high', 'low')
    return result


# %% These scoring helpers receive final votes after prediction has completed.
def district_scores(features, observed, predicted, final):
    """Score only missing ordinary units; no reported counts dilute the error.

    District share MAE is averaged over eight named parties, then weighted by
    final valid votes. Volume WAPE sums absolute district volume errors divided
    by final valid votes. Group aggregate shares keep Other in the denominator.
    Empty groups deliberately return NaN errors (never perfect zero errors).
    """
    observed, predicted, final = map(np.asarray, (observed, predicted, final))
    if any(a.shape != (len(features), len(PARTIES)) for a in (observed, predicted, final)):
        raise ValueError('District vote arrays must match feature rows and parties')
    if not np.isfinite(predicted).all() or not np.isfinite(final).all() or (predicted < 0).any() or (final < 0).any():
        raise ValueError('Prediction and scoring votes must be finite and nonnegative')
    known = np.isfinite(observed).all(axis=1)
    missing = ~known & features.kind.eq('ordinary').to_numpy()
    rows, party_rows = [], []
    for source, label in GROUPS.items():
        mask = missing & baseline_groups(features).eq(source).to_numpy()
        row = {'baseline_source': source, 'group': label, 'districts': int(mask.sum()),
               'electorate': float(features.loc[mask, 'electorate'].sum()), 'final_votes': 0.,
               'district_share_mae_pp': np.nan, 'volume_wape_pct': np.nan,
               'volume_bias_pct': np.nan, 'aggregate_share_mae_pp': np.nan,
               'aggregate_share_max_abs_error_pp': np.nan}
        if mask.any():
            p, y = predicted[mask], final[mask]
            pt, yt = p.sum(axis=1), y.sum(axis=1)
            if (pt <= 0).any() or (yt <= 0).any():
                raise ValueError('Scored districts require positive valid-vote volumes')
            errors = 100 * (p[:, NAMED_IDS] / pt[:, None] - y[:, NAMED_IDS] / yt[:, None])
            aggregate_errors = 100 * (p.sum(axis=0) / p.sum() - y.sum(axis=0) / y.sum())
            row.update(final_votes=float(y.sum()),
                       district_share_mae_pp=float(np.average(np.abs(errors).mean(axis=1), weights=yt)),
                       volume_wape_pct=float(100 * np.abs(pt - yt).sum() / yt.sum()),
                       volume_bias_pct=float(100 * (pt.sum() / yt.sum() - 1)),
                       aggregate_share_mae_pp=float(np.abs(aggregate_errors[NAMED_IDS]).mean()),
                       aggregate_share_max_abs_error_pp=float(np.abs(aggregate_errors[NAMED_IDS]).max()))
            for j, party in enumerate(NAMED):
                party_rows.append({'baseline_source': source, 'party': party,
                                   'district_share_mae_pp': float(np.average(np.abs(errors[:, j]), weights=yt)),
                                   'aggregate_share_error_pp': float(aggregate_errors[PARTIES.index(party)]),
                                   'aggregate_predicted_share_pct': float(100 * p[:, PARTIES.index(party)].sum() / p.sum()),
                                   'aggregate_actual_share_pct': float(100 * y[:, PARTIES.index(party)].sum() / y.sum())})
        rows.append(row)
    return rows, party_rows


def constituency_scores(predicted, final, constituencies, allocation, fixture, exposure, simulation):
    exposure = exposure.set_index('constituency')
    codes = simulation.constituency_codes
    intervals, rows = [], []
    shares = simulation.vote_draws / simulation.vote_draws.sum(axis=2, keepdims=True) * 100
    lower, upper = np.quantile(shares, [.05, .95], axis=0)
    for c, code in enumerate(codes):
        mask = np.asarray(constituencies) == code
        p, y = predicted[mask].sum(axis=0), final[mask].sum(axis=0)
        point, actual = 100 * p / p.sum(), 100 * y / y.sum()
        metadata = {'constituency': code, 'constituency_name': fixture['names'][code],
                    'fallback_electorate_fraction': float(exposure.loc[code, 'fallback_electorate_fraction']),
                    'exposure_group': exposure.loc[code, 'exposure_group']}
        for party in NAMED:
            j = PARTIES.index(party)
            seat = allocation['constituencies'][code]
            truth = fixture['constituencies'][code]
            rows.append({**metadata, 'party': party, 'share_error_pp': float(point[j] - actual[j]),
                         'predicted_share_pct': float(point[j]), 'actual_share_pct': float(actual[j]),
                         'predicted_seats': seat['total'].get(party, 0), 'actual_seats': truth['total'].get(party, 0),
                         'seat_error': seat['total'].get(party, 0) - truth['total'].get(party, 0),
                         'fixed_seat_error': seat['fixed'].get(party, 0) - truth['fixed'].get(party, 0),
                         'adjustment_seat_error': seat['adjustment'].get(party, 0) - truth['adjustment'].get(party, 0)})
            intervals.append({**metadata, 'party': party, 'nominal': .90,
                              'lower_pct': float(lower[c, j]), 'upper_pct': float(upper[c, j]),
                              'width_pp': float(upper[c, j] - lower[c, j]), 'actual_pct': float(actual[j]),
                              'covered': bool(lower[c, j] <= actual[j] <= upper[c, j]),
                              'distance_outside_pp': float(max(lower[c, j] - actual[j], actual[j] - upper[c, j], 0))})
    return rows, intervals


def summarize(groups, seats, intervals):
    group_rows = []
    for source, frame in groups.loc[groups.districts.gt(0)].groupby('baseline_source'):
        weights = frame.final_votes
        group_rows.append({'baseline_source': source, 'group': GROUPS[source], 'scored_cases': len(frame),
                           'district_share_mae_pp': float(np.average(frame.district_share_mae_pp, weights=weights)),
                           'volume_wape_pct': float(np.average(frame.volume_wape_pct, weights=weights)),
                           'mean_aggregate_share_mae_pp': float(frame.aggregate_share_mae_pp.mean())})
    seat_summary = (seats.assign(abs_seat_error=seats.seat_error.abs(), abs_share_error=seats.share_error_pp.abs())
                    .groupby(['scenario', 'fraction', 'exposure_group'], as_index=False)
                    .agg(party_seat_mae=('abs_seat_error', 'mean'), max_party_seat_error=('abs_seat_error', 'max'),
                         share_mae_pp=('abs_share_error', 'mean'), scored_party_constituencies=('party', 'size')))
    coverage = (intervals.groupby(['scenario', 'fraction', 'exposure_group'], as_index=False)
                .agg(coverage=('covered', 'mean'), mean_width_pp=('width_pp', 'mean'),
                     worst_miss_pp=('distance_outside_pp', 'max'), scored_party_constituencies=('party', 'size')))
    return pd.DataFrame(group_rows), seat_summary, coverage


def draw_charts(groups, seats, coverage, exposure):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'district_mapping': '#3079a0', 'municipal_fallback': '#be6741'}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout='constrained')
    for source, color in colors.items():
        archived = groups[(groups.scenario == 'archived_report_time') & (groups.baseline_source == source)]
        axes[0].plot(archived.fraction * 100, archived.district_share_mae_pp, marker='o', color=color, label=GROUPS[source])
        axes[1].plot(archived.fraction * 100, archived.aggregate_share_mae_pp, marker='o', color=color)
    axes[0].set(title='Unreported district share error', ylabel='Vote-weighted mean absolute error (pp)')
    axes[1].set(title='Unreported group aggregate error', ylabel='Mean absolute party-share error (pp)')
    by_constituency = (seats[seats.scenario.eq('archived_report_time') & seats.fraction.eq(.50)]
                      .assign(absolute=seats.seat_error.abs()).groupby('constituency').absolute.mean())
    x = exposure.set_index('constituency').loc[by_constituency.index, 'fallback_electorate_fraction']
    axes[2].scatter(100 * x, by_constituency, color='#71598e', alpha=.8)
    for code in by_constituency.nlargest(3).index:
        axes[2].annotate(code, (100 * x.loc[code], by_constituency.loc[code]), xytext=(4, 5), textcoords='offset points')
    axes[2].set(title='Constituency seat error at 50% reporting', xlabel='Electorate using municipal fallback (%)',
                ylabel='Mean absolute party seat error')
    for ax in axes[:2]:
        ax.set(xlabel='Ordinary districts reported (%)', xlim=(0, 100))
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle('2018 → 2022 diagnostic · archived report/correction order · frozen model', fontsize=14)
    fig.savefig(OUT / 'district-and-seat-errors.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for exposure_group, color in [('low', '#3079a0'), ('high', '#be6741')]:
        archived = coverage[(coverage.scenario == 'archived_report_time') & (coverage.exposure_group == exposure_group)]
        axes[0].plot(100 * archived.fraction, 100 * archived.coverage, marker='o', color=color,
                     label=EXPOSURE_LABELS[exposure_group])
        axes[1].plot(100 * archived.fraction, archived.mean_width_pp, marker='o', color=color)
    axes[0].axhline(90, color='#777777', linestyle='--', linewidth=1)
    axes[0].set(title='Final constituency party-share interval coverage', ylabel='Outcomes covered (%)', ylim=(0, 103))
    axes[1].set(title='Mean 90% interval width', ylabel='Percentage points')
    for ax in axes:
        ax.set(xlabel='Ordinary districts reported (%)', xlim=(0, 100))
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle('Unchanged approximate Bayesian simulation · one election, descriptive coverage only', fontsize=13)
    fig.savefig(OUT / 'constituency-intervals.png', dpi=160)
    plt.close(fig)


# %% The report explicitly separates local errors from aggregate seat outcomes.
def render_report(groups, seats, intervals, exposure, current_exposure, manifest):
    helpers = history_helpers()
    table = helpers.markdown_table
    group_summary, seat_summary, coverage = summarize(groups, seats, intervals)
    group_summary.to_csv(OUT / 'group-summary.csv', index=False)
    seat_summary.to_csv(OUT / 'constituency-summary.csv', index=False)
    coverage.to_csv(OUT / 'interval-summary.csv', index=False)
    draw_charts(groups, seats, coverage, exposure)
    counts = exposure.groupby('exposure_group').size().to_dict()
    overall_coverage = intervals.groupby('exposure_group').agg(coverage=('covered', 'mean'), width=('width_pp', 'mean'))
    archived_rows = []
    for fraction in [.01, .10, .50, .90, 1.]:
        a = groups[groups.scenario.eq('archived_report_time') & groups.fraction.eq(fraction)].set_index('baseline_source')
        row = {'Reported': f'{fraction:.0%}'}
        for source, label in GROUPS.items():
            v = a.loc[source]
            row[label + ' district MAE'] = '—' if not v.districts else f'{v.district_share_mae_pp:.2f} pp'
            row[label + ' group MAE'] = '—' if not v.districts else f'{v.aggregate_share_mae_pp:.2f} pp'
        archived_rows.append(row)
    worst_group = groups[groups.districts.gt(0)].nlargest(6, 'aggregate_share_mae_pp')[
        ['scenario', 'fraction', 'group', 'districts', 'district_share_mae_pp', 'aggregate_share_mae_pp', 'volume_wape_pct']]
    worst_seats = (seats[seats.fraction.ge(.50)].assign(absolute=seats.seat_error.abs())
                   .sort_values(['absolute', 'fraction'], ascending=False).head(8)[
                       ['scenario', 'fraction', 'constituency_name', 'party', 'seat_error', 'fallback_electorate_fraction']])
    worst_intervals = intervals.nlargest(8, 'distance_outside_pp')[
        ['scenario', 'fraction', 'constituency_name', 'party', 'exposure_group', 'lower_pct', 'upper_pct', 'actual_pct', 'distance_outside_pp']]
    current_top = current_exposure.nlargest(8, 'fallback_electorate_fraction')[[
        'constituency', 'constituency_name', 'fallback_districts', 'districts', 'fallback_electorate_fraction']]
    scalar_format = {c: (lambda v: f'{v:.2f}') for c in ['district_share_mae_pp', 'aggregate_share_mae_pp',
        'volume_wape_pct', 'mean_aggregate_share_mae_pp', 'lower_pct', 'upper_pct', 'actual_pct', 'distance_outside_pp']}
    scalar_format.update({'fraction': lambda v: f'{v:.0%}', 'fallback_electorate_fraction': lambda v: f'{v:.1%}'})
    mapped = group_summary.set_index('baseline_source').loc['district_mapping']
    fallback = group_summary.set_index('baseline_source').loc['municipal_fallback']
    local_ratio = fallback.district_share_mae_pp / mapped.district_share_mae_pp
    coverage_text = '; '.join(f'{EXPOSURE_LABELS[g]}: **{row.coverage:.1%}** covered, mean width **{row.width:.2f} pp**'
                              for g, row in overall_coverage.iterrows())
    archived_mid = seats[seats.scenario.eq('archived_report_time') & seats.fraction.eq(.50)].copy()
    const_mid = archived_mid.assign(absolute=archived_mid.seat_error.abs()).groupby('constituency').agg(
        mae=('absolute', 'mean'), exposure=('fallback_electorate_fraction', 'first'))
    correlation = const_mid.mae.corr(const_mid.exposure)
    exposure_share = exposure.fallback_electorate.sum() / exposure.electorate.sum()
    current_share = current_exposure.fallback_electorate.sum() / current_exposure.electorate.sum()
    archived_coverage = coverage[coverage.scenario.eq('archived_report_time')].copy()
    archived_coverage['coverage'] = archived_coverage.coverage.map(lambda x: f'{x:.1%}')
    archived_coverage['mean_width_pp'] = archived_coverage.mean_width_pp.map(lambda x: f'{x:.2f}')
    archived_coverage['fraction'] = archived_coverage.fraction.map(lambda x: f'{x:.0%}')
    archived_coverage = archived_coverage[['fraction', 'exposure_group', 'coverage', 'mean_width_pp']]
    max_late_seat_error = int(seats.loc[seats.fraction.ge(.50), 'seat_error'].abs().max())
    minimum_coverage = coverage.coverage.min()
    report = f'''# Changed-district readiness diagnostic

The frozen regression has larger **individual-district** share errors for municipal-fallback units: **{fallback.district_share_mae_pp:.2f} pp** versus **{mapped.district_share_mae_pp:.2f} pp** for comparable units ({local_ratio:.2f}×), pooling all nonempty scenario/checkpoint groups with final-valid-vote weights. This is a diagnostic comparison, not a causal estimate of the effect of changed boundaries. Municipal fallback loses local detail; errors can offset when aggregated. Group aggregate errors and complete constituency seat forecasts are therefore scored separately below.

No point-model, uncertainty or production-input settings changed. All {manifest['cases']} cases use the frozen 2018 background, revealed preliminary 2022 returns and the existing reporting-order scenarios. Final 2022 counts enter only the scoring helpers. The original point model was selected earlier on this same election, so this is **not an untouched test election and not evidence of 2026 calibration**.

**Readiness decision:** retain the current point model and uncertainty settings. The larger local fallback errors do not translate into evidence here that the constituency ranges need widening: coverage is at least **{minimum_coverage:.1%}** in every scenario/checkpoint/exposure group, and from 50% reporting onward the largest party/constituency seat error across the five scenarios is **{max_late_seat_error} seat**. This supports keeping the tested configuration for launch; it does not prove that 2026 uncertainty is calibrated. The constituency misses below remain useful diagnostics, especially early in the count.

## Scope and scoring

- Historical replay: **{int(exposure.districts.sum()):,} ordinary districts**, including **{int(exposure.fallback_districts.sum()):,} municipal-fallback districts**, covering **{exposure_share:.2%}** of the ordinary electorate.
- Municipal fallback pools both residual-municipality baselines and the two 2022 units that need the full municipality average; neither is omitted from scoring.
- Current 2026 inputs: **{int(current_exposure.districts.sum()):,} ordinary districts**, including **{int(current_exposure.fallback_districts.sum()):,} municipal-fallback districts**, covering **{current_share:.2%}** of the ordinary electorate. The historical replay has more fallback exposure, but that alone does not guarantee conservative future errors.
- District and fallback-group scores exclude reported ordinary districts and all late collection pools. At 100% ordinary reporting no such units remain: their errors are missing, not zero.
- District share MAE: mean absolute share error across the eight named parties, weighted across unreported districts by final valid votes. Shares retain all nine categories, including Other, in their denominator.
- Volume WAPE: sum of absolute unreported district valid-vote volume errors divided by their final valid-vote total. Signed group volume bias is also saved.
- Group aggregate share MAE: pool unreported votes within each baseline group, then average absolute share errors across eight named parties. It measures cancellation within each group; it is not the national forecast error.
- Constituency seat errors compare full frozen-model forecasts with final official 2022 constituency allocations, using known 2022 membership and fixed seats. They include reported preliminary votes, outstanding ordinary districts and late pools; seat errors also reflect national thresholds and adjustment mandates.
- Uncertainty: **{manifest['draws_per_checkpoint']} joint draws per case**, the unchanged full configuration. Constituencies are assigned **before scoring** to low exposure (at most one-third fallback electorate; {counts.get('low', 0)} constituencies) or high exposure (over one-third; {counts.get('high', 0)} constituencies). Coverage and widths concern final constituency party vote shares, not individual district shares or constituency seats.
- All five reporting orders and all eight existing fractions are included. Archived order uses surviving report/correction timestamps, which may differ from first arrival; the other four are synthetic stress scenarios with seed 100. Late pools remain outstanding at 100% ordinary reporting.

## District errors

{table(group_summary, scalar_format)}

The pooled district MAE/WAPE weight each case by its outstanding final votes. Mean group aggregate MAE weights the nonempty cases equally. Repeated districts and checkpoints are dependent, so these pooled summaries are descriptive.

Selected archived-order checkpoints:

{table(pd.DataFrame(archived_rows))}

![District and seat errors](district-and-seat-errors.png)

Worst group aggregate errors across all scenarios and stages:

{table(worst_group, scalar_format)}

## Constituency seats and fallback exposure

At 50% reporting in archived order, the descriptive Pearson correlation between constituency fallback-electorate exposure and mean absolute party seat error is **{correlation:.2f}**. This single cross-sectional association is not a causal effect: constituencies differ in size, political mix, reporting progress and national adjustment-mandate effects. The raw table retains total, fixed and adjustment seat errors for every party/constituency/case. We do not halve local absolute seat errors into “seats to move”, because adjustment mandates can change a constituency's total seats.

Largest absolute party/constituency seat errors from 50% reporting onward:

{table(worst_seats, scalar_format)}

## Unchanged uncertainty: final constituency shares

Across all scenarios and stages, {coverage_text}. Each party/constituency/case receives equal weight. These outcomes are strongly dependent; nominal 90% coverage is a diagnostic reference, not an independent calibration test. The draw count also makes tail quantiles noisy.

{table(archived_coverage)}

![Constituency interval diagnostic](constituency-intervals.png)

Largest interval misses (positive distance indicates an uncovered final share):

{table(worst_intervals, scalar_format)}

## Current 2026 concentration

{table(current_top, scalar_format)}

The 2026 table describes frozen pre-election input exposure, not forecast error. No 2026 final results are used. Inspect these constituencies together with first production-feed alignment and observed reporting balance; do not mechanically inflate all election-level uncertainty by the district-error ratio. Whether an additional fallback-specific residual scale is useful requires examining the misses above and the national/constituency consequences; one election cannot identify a reliable new variance multiplier.

## Reproducibility and limits

Completed {manifest['completed_at_utc']} in {manifest['runtime_seconds']:.1f} seconds. `config.json` records source/code/lock hashes, the original model lock, unchanged uncertainty settings, seeds, intervals and the predeclared exposure cutoff. CSVs preserve each case's group errors, party-specific group errors, constituency shares/seats, interval bounds and summaries. Historical and current constituency exposure tables are separate. This is a local diagnostic; it does not alter the live report, publish data, refit settings against final outcomes or establish independent calibration.
'''
    (OUT / 'report.md').write_text(report)


# %% Fit, simulate, then score. Future outcomes are never supplied to construction.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draws', type=int, default=256)
    args = parser.parse_args(argv)
    if args.draws < 20:
        parser.error('Use at least 20 draws for interval scoring')
    started = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    helpers = history_helpers()
    features, preliminary, final, fixture = helpers.load_history()
    selected, options, lock = frozen_model(ROOT)
    model = SwingRegression(features, **options)
    exposure = exposure_table(features)
    exposure['constituency_name'] = exposure.constituency.map(fixture['names'])
    current, current_manifest = load_inputs(ROOT)
    current_exposure = exposure_table(current)
    current_exposure['constituency_name'] = current_exposure.constituency.map(
        current.groupby('constituency').constituency_name.first())
    exposure.to_csv(OUT / 'constituency-exposure-2022.csv', index=False)
    current_exposure.to_csv(OUT / 'constituency-exposure-2026.csv', index=False)
    config = asdict(UncertaintyConfig())
    paths = ['scripts/check-district-changes.py', 'scripts/backtest-uncertainty.py',
             'val2026/forecast_model.py', 'val2026/uncertainty.py', 'val2026/seats.py', 'val2026/replay.py',
             'outputs/selected-model.json', 'data/processed/backtest-2018-2022.csv',
             'data/processed/2022-report-times.csv', 'data/seats/history/2022.json',
             'data/seats/history/backtest-2022-constituencies.csv', 'data/inputs/latest-success.json',
             current_manifest['baseline_path'], 'uv.lock']
    manifest = {'model': selected, 'model_options': options, 'model_lock': lock, 'uncertainty_config': config,
                'scenarios': SCENARIOS, 'fractions': FRACTIONS, 'draws_per_checkpoint': args.draws,
                'arrival_seed_rule': '0 for archived report times; 100 for all synthetic orders',
                'draw_seed_rule': '2026000 + SCENARIOS index * 10000 + round(fraction * 1000)',
                'exposure_cutoff': EXPOSURE_CUTOFF, 'exposure_basis': 'ordinary pre-election electorate',
                'district_score_mask': 'ordinary AND not fully observed; split by baseline_source',
                'municipal_fallback_sources': ['municipality_residual', 'municipality_average'],
                'scored_parties': NAMED, 'share_denominator_parties': PARTIES, 'nominal_interval': .90,
                'final_outcomes_usage': 'validation and scoring only; excluded from prediction and simulation',
                'input_snapshot_2026': current_manifest['snapshot_id'],
                'sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}}
    write_json(OUT / 'config.json', manifest)
    all_groups, all_group_parties, all_seats, all_intervals, diagnostics = [], [], [], [], []
    for scenario in SCENARIOS:
        arrival_seed = 0 if scenario == 'archived_report_time' else 100
        order = arrival_order(features, scenario, arrival_seed)
        for fraction in FRACTIONS:
            case_started = perf_counter()
            keys = {'scenario': scenario, 'fraction': fraction, 'arrival_seed': arrival_seed}
            observed = reveal(preliminary, order, fraction)
            predicted = model.fit_predict(observed)
            seed = 2026000 + SCENARIOS.index(scenario) * 10000 + round(fraction * 1000)
            simulation = simulate_constituency_votes(model, observed, predicted=predicted,
                                                     draws=args.draws, seed=seed, config=config)
            allocation = allocate_riksdag(helpers.aggregate_votes(predicted, features.constituency),
                                          fixture['fixed_seats'], tie_seed=seed)
            groups, group_parties = district_scores(features, observed, predicted, final)
            seats, intervals = constituency_scores(predicted, final, features.constituency,
                                                   allocation, fixture, exposure, simulation)
            all_groups.extend({**keys, **r} for r in groups)
            all_group_parties.extend({**keys, **r} for r in group_parties)
            all_seats.extend({**keys, **r} for r in seats)
            all_intervals.extend({**keys, **r} for r in intervals)
            diagnostics.append({**keys, 'draw_seed': seed, 'seconds': perf_counter() - case_started,
                                'point_allocation_ties': allocation['ties'],
                                'simulation': simulation.diagnostics})
            print(f'{scenario} {fraction:.0%}: ' + ', '.join(
                f'{r["group"]}: {r["district_share_mae_pp"]:.2f} pp ({r["districts"]} outstanding)' for r in groups), flush=True)
    groups, seats, intervals = map(pd.DataFrame, (all_groups, all_seats, all_intervals))
    groups.to_csv(OUT / 'district-group-cases.csv', index=False)
    pd.DataFrame(all_group_parties).to_csv(OUT / 'district-group-parties.csv', index=False)
    seats.to_csv(OUT / 'constituency-party-cases.csv', index=False)
    intervals.to_csv(OUT / 'constituency-share-intervals.csv', index=False)
    write_json(OUT / 'diagnostics.json', diagnostics)
    manifest.update(cases=len(diagnostics), runtime_seconds=perf_counter() - started,
                    completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(OUT / 'config.json', manifest)
    render_report(groups, seats, intervals, exposure, current_exposure, manifest)
    print(f'Completed {len(diagnostics)} cases; {OUT / "report.md"}', flush=True)
    return groups, seats, intervals


# %%
if __name__ == '__main__':
    groups, seats, intervals = main()
