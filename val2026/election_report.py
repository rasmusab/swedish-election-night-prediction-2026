"""One bounded update, persistent report state, and presentation helpers."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from val2026.election_feed import BASE_URLS, Collector, FeedError, atomic_write, json_bytes, source_freshness
from val2026.election_forecast import ROOT, load_inputs, load_snapshot, evaluate_snapshot
from val2026.locking import update_lock
from val2026.forecast_uncertainty import DEFAULT_DRAWS

STOCKHOLM = ZoneInfo('Europe/Stockholm')
PARTY_COLORS = {'V': '#a82332', 'S': '#d44545', 'MP': '#70a63d', 'C': '#258653',
                'M': '#3276ab', 'L': '#66add1', 'KD': '#55548b', 'SD': '#e7bd46', 'ÖVR': '#929aa4'}


def phase_at(now):
    local = now.astimezone(STOCKHOLM)
    if local < datetime(2026, 9, 13, 20, tzinfo=STOCKHOLM):
        return 'before_polls_close'
    if local < datetime(2026, 9, 14, 8, tzinfo=STOCKHOLM):
        return 'election_night'
    if local < datetime(2026, 9, 16, tzinfo=STOCKHOLM):
        return 'preliminary_pause'
    return 'late_preliminary_count'


def run_cycle(root=ROOT, environment='production', *, collect=True, now=None, collector=None, simulation_draws=DEFAULT_DRAWS):
    """Called under update_lock by the notebook or export command.

    Final counting is never selected here. A failed collect/schema/model run
    preserves the last model success and labels it with its original timestamp.
    Repeated valid snapshots may be re-fit, but history only records new inputs.
    """
    if environment not in ('production', 'rehearsal'):
        raise ValueError('Use an explicit production or rehearsal environment')
    now = now or datetime.now(timezone.utc)
    output = root / 'outputs' / f'latest-{environment}'
    output.mkdir(parents=True, exist_ok=True)
    state = {'generated_at_utc': now.isoformat(), 'environment': environment,
             'test': environment == 'rehearsal', 'count': 'preliminary',
             'phase': phase_at(now), 'health': 'waiting', 'estimate_state': 'waiting',
             'parties': [], 'message': 'Waiting for the official preliminary result feed.'}
    cache = root / 'data/collector' / environment / 'preliminary'
    # Server retry delays begin when a response arrives, not at report start.
    check = (collector or Collector(root, environment=environment)).check() if collect else (
        json.loads((cache / 'latest-check.json').read_bytes()) if (cache / 'latest-check.json').exists() else {})
    state['latest_check'] = check
    state['offline'] = not collect
    error = None
    success = None
    try:
        if check.get('status') == 'error':
            raise FeedError(check.get('error', 'Latest collection failed'))
        if environment == 'production' and state['phase'] == 'before_polls_close':
            # Cache and validate a pre-published production roster if available,
            # but never display numerical election estimates before polls close.
            if (cache / 'latest-success.json').exists():
                features, inputs = load_inputs(root)
                receipt, source = load_snapshot(root, environment)
                from val2026.election_forecast import resolve_roster
                resolve_roster(root, environment, features, source, receipt)
            state['message'] = 'Polls close on 13 September at 20:00 Swedish time. No forecast is published before then.'
        elif (cache / 'latest-success.json').exists():
            features, inputs = load_inputs(root)
            receipt, source = load_snapshot(root, environment)
            success, districts = evaluate_snapshot(root, environment, features, inputs, receipt, source,
                                                   now=now, simulation_draws=simulation_draws)
            state.update(success)
            state['health'] = 'ok'
            if districts is not None:
                # The atomic success bundle is the fallback; CSVs are convenience exports.
                atomic_write(output / 'prediction.json', json_bytes(success))
                atomic_write(output / 'districts.csv', districts.to_csv(index=False).encode())
                atomic_write(output / 'parties.csv', pd.DataFrame(success['parties']).to_csv(index=False).encode())
                seat_rows = [{'constituency': code, 'party': party,
                              'fixed_seats': values['fixed'][party],
                              'adjustment_seats': values['adjustment'][party], 'seats': seats}
                             for code, values in success['seat_allocation']['constituencies'].items()
                             for party, seats in values['total'].items()]
                atomic_write(output / 'constituency-seats.csv', pd.DataFrame(seat_rows).to_csv(index=False).encode())
                uncertainty = success['uncertainty']
                draw_table = pd.DataFrame(uncertainty['seat_draws'], columns=uncertainty['seat_draw_parties'])
                for index, bloc in enumerate(uncertainty['blocs'], 1):
                    draw_table[f'bloc_{index}'] = draw_table[bloc['parties']].sum(axis=1)
                atomic_write(output / 'seat-draws.csv', draw_table.to_csv(index=False).encode())
                update_history(output, success)
        else:
            state['message'] = 'The official preliminary feed has not supplied a verified snapshot yet.'
    except (FeedError, ValueError, OSError, KeyError, TypeError) as exc:
        error = str(exc)
        previous = output / 'prediction.json'
        if previous.exists() and not (environment == 'production' and phase_at(now) == 'before_polls_close'):
            saved = json.loads(previous.read_bytes())
            if saved.get('environment') == environment and saved.get('test') is (environment == 'rehearsal'):
                state.update(saved)
                state['retained_previous_estimate'] = True
        state.update(health='update_failed', error=error,
                     message='The latest update failed. Any displayed estimate is the last successful one.')
        if (not state['parties'] and check.get('http_status') == 404
                and check.get('failed_url') == BASE_URLS['production'] + 'index.md5'
                and environment == 'production' and phase_at(now) == 'before_polls_close'):
            state.update(health='waiting', message='Waiting for Valmyndigheten to publish the production feed.')
    state.update(generated_at_utc=now.isoformat(), latest_check=check, offline=not collect, phase=phase_at(now))
    if state.get('source_updated_at'):
        state.update(source_freshness(state['source_updated_at'], now, 15))
        if state['health'] == 'ok' and state['source_freshness'] != 'recent':
            state['health'] = 'stale'
    if not collect and state['health'] == 'ok':
        state['health'] = 'cached'
    state['history'] = json.loads((output / 'history.json').read_bytes()) if (output / 'history.json').exists() else []
    atomic_write(output / 'report-state.json', json_bytes(state))
    return state


def update_history(output, success):
    path = output / 'history.json'
    history = json.loads(path.read_bytes()) if path.exists() else []
    identity = [success['archive_sha256'], success['baseline_sha256'], success['model_lock']['lock_id'],
                success['fixed_seats_sha256'], success['uncertainty']['draws'], success['uncertainty']['seed']]
    if history and history[-1]['identity'] == identity:
        return
    history.append({'identity': identity, 'source_updated_at': success['source_updated_at'],
                    'estimate_generated_at_utc': success['estimate_generated_at_utc'],
                    'ordinary_districts_observed': success['ordinary_districts_observed'],
                    'shares': {p['party']: p['forecast_share_pct'] for p in success['parties']},
                    'seats': {p['party']: p['forecast_seats'] for p in success['parties']},
                    'blocs': {b['name']: b['seats'] for b in success['blocs']},
                    'bloc_intervals': {b['name']: b['uncertainty'] for b in success['blocs']}})
    atomic_write(path, json_bytes(history))


def timestamp(value):
    if not value:
        return 'Not available'
    parsed = datetime.fromisoformat(value)
    parsed = parsed.replace(tzinfo=STOCKHOLM) if parsed.tzinfo is None else parsed
    return parsed.astimezone(STOCKHOLM).strftime('%d %b %H:%M:%S %Z')


def header_html(state):
    test = state['test']
    label = 'REHEARSAL · ARTIFICIAL RESULTS' if test else 'SWEDEN · RIKSDAG 2026'
    badge = {'ok': 'Feed checked', 'waiting': 'Waiting for results', 'stale': 'Source has not updated recently',
             'update_failed': 'Update failed · previous estimate retained', 'cached': 'Cached data · no live check'}[state['health']]
    if state['health'] == 'update_failed' and not state.get('retained_previous_estimate'):
        badge = 'Update failed · no estimate available'
    if state.get('source_freshness') == 'future_timestamp':
        badge += ' · Source timestamp is in the future'
    if state.get('offline'):
        badge += ' · Offline run'
    metrics = [
        ('Districts reporting', f"{state.get('ordinary_districts_observed', 0):,} / {state.get('baseline_ordinary_districts', '—'):,}" if 'baseline_ordinary_districts' in state else '—'),
        ('Expected ordinary votes covered', f"{state.get('expected_vote_weight_reported_pct', 0):.1f}%" if 'expected_vote_weight_reported_pct' in state else '—'),
        ('Counties reporting', f"{state.get('counties_reporting', 0)} / {state.get('counties_total', '—')}"),
        ('Reported valid votes', f"{state.get('raw_valid_votes', 0):,}")]
    cards = ''.join(f'<div class="metric"><span>{escape(k)}</span><strong>{escape(v)}</strong></div>' for k,v in metrics)
    phase_messages = {
        'before_polls_close': 'Election day is 13 September. Polls close at 20:00 Swedish time.',
        'election_night': 'Election-night preliminary count. Outstanding ordinary districts and late votes are estimated.',
        'preliminary_pause': 'The preliminary feed normally pauses after election night. The separate final recount is not included here.',
        'late_preliminary_count': 'Preliminary collection count: late votes are being added. The separate final recount is not included here.'}
    warning = '<p class="test-note">These are artificial rehearsal votes. They say nothing about the likely 2026 result.</p>' if test else ''
    error = f'<details><summary>Update details</summary><p>{escape(state["error"])}</p></details>' if state.get('error') else ''
    roster_note = '<p class="test-note">The complete live district roster is not available yet. Late-vote pools remain open until it has been verified.</p>' if state.get('roster_source', {}).get('status') == 'pending_full_feed' else ''
    early = '<p class="test-note">EARLY ESTIMATE · Fewer than 300 districts. The first districts can be strongly unrepresentative.</p>' if state.get('estimate_state') == 'early' else ''
    return f'''<style>
body {{color:#182536;background:#f4f6f9;font-family:system-ui,sans-serif}} .jp-Notebook {{max-width:1120px;margin:auto;background:white;padding:28px!important}}
.jp-InputArea,.jp-InputPrompt,.jp-OutputPrompt {{display:none!important}} h1 {{font-size:2.4rem!important;line-height:1.1!important}} h2 {{margin-top:1.6rem!important}}
.eyebrow {{font-size:.8rem;letter-spacing:.14em;font-weight:700;color:#46576b}} .badge {{display:inline-block;padding:8px 12px;background:#e9eef5;border-radius:4px}}
.test-note {{border-left:4px solid #b36a12;background:#fff4df;padding:12px}} .metrics {{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}
.metric {{border-top:2px solid #283e59;padding:12px 0}} .metric span {{display:block;color:#586778;font-size:.85rem}} .metric strong {{display:block;font-size:1.75rem;margin-top:8px}}
.bloc-cards {{display:grid;grid-template-columns:repeat(2,1fr);gap:24px;margin:20px 0}} .bloc-card {{padding:18px;background:#f3f6fa;border-radius:6px}} .bloc-card strong {{font-size:2.5rem;display:block;line-height:1.4}} .bloc-card small {{display:block;color:#586778}} .seat-total {{font-weight:700}}
.majority-probability {{margin-top:16px;padding-top:12px;border-top:1px solid #d5dfe9}} .majority-probability b {{display:block;font-size:1.8rem;line-height:1.4}}
table {{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}} th,td {{padding:9px!important;text-align:right!important;border-bottom:1px solid #e5e9ef}} th:first-child,td:first-child {{text-align:left!important}}
.times {{font-size:.85rem;color:#586778;line-height:1.7}} img {{max-width:100%;height:auto}} @media(max-width:650px) {{.metrics{{grid-template-columns:repeat(2,1fr)}} .jp-Notebook{{padding:10px!important}} h1{{font-size:1.8rem!important}}}}
</style><div class="eyebrow">{label}</div><h1>Election-night estimate</h1>
<div class="badge" role="status">{escape(badge)}</div>{warning}{early}{roster_note}<p>{escape(state['message'])}</p>
<p>{phase_messages[state['phase']] if not test else 'Rehearsal integration run of the preliminary reporting workflow.'}</p>{error}
<div class="metrics">{cards}</div><div class="times">
Source updated: {timestamp(state.get('source_updated_at'))}<br>
Latest fetch attempt: {timestamp(state.get('latest_check', {}).get('checked_at_utc'))}<br>
Estimate computed: {timestamp(state.get('estimate_generated_at_utc'))}<br>
Page generated: {timestamp(state['generated_at_utc'])}</div>'''


def party_table(state):
    if not state['parties']:
        return '<p>No numerical estimate is available yet.</p>'
    table = pd.DataFrame(state['parties']).set_index('party')
    cols = []
    if 'forecast_seats' in table:
        cols += ['forecast_seats']
    details = ''
    if 'seat_lower_90' in table:
        for interval in (50, 90):
            column = f'{interval}% seat range'
            table[column] = [f'{int(lo)}–{int(hi)}' for lo, hi in
                             zip(table[f'seat_lower_{interval}'], table[f'seat_upper_{interval}'])]
            cols += [column]
        details = ('<details><summary>Fixed and adjustment seats in the point projection</summary>' +
                   table[['forecast_seats', 'fixed_seats', 'adjustment_seats']].rename(columns={
                       'forecast_seats': 'Seats', 'fixed_seats': 'Fixed', 'adjustment_seats': 'Adjustment'
                   }).to_html(border=0) + '</details>')
    if 'forecast_share_pct' in table:
        cols += ['forecast_share_pct']
    cols += ['raw_reported_share_pct']
    table = table[cols].rename(columns={'raw_reported_share_pct': 'Counted votes %',
        'forecast_share_pct': 'Projected votes %', 'forecast_seats': 'Seats',
        'fixed_seats': 'Fixed', 'adjustment_seats': 'Adjustment'})
    title = 'Seats by party' if 'seat_allocation' in state else 'Counted vote shares'
    return f'<h2>{title}</h2>' + table.to_html(float_format=lambda x: f'{x:.2f}', na_rep='—', border=0) + details


def seat_summary_html(state):
    if not state.get('seat_allocation'):
        return '<h2>Projected Riksdag seats</h2><p>Seat estimates will appear once enough districts have reported.</p>'
    cards = []
    probabilities = {b['name']: b['majority_probability']
                     for b in state.get('uncertainty', {}).get('blocs', [])
                     if b.get('majority_probability') is not None}
    for bloc in state['blocs']:
        distance = bloc['seats'] - bloc['majority_threshold']
        detail = (f'{distance} above the majority threshold' if distance > 0 else
                  'At the majority threshold' if distance == 0 else f'{-distance} short of a majority')
        interval = bloc.get('uncertainty')
        ranges = (f'<small>50% range: {interval["lower_50"]}–{interval["upper_50"]} · '
                  f'90% range: {interval["lower_90"]}–{interval["upper_90"]}</small>' if interval else '')
        probability = ''
        if bloc['name'] in probabilities:
            p = probabilities[bloc['name']]
            # Avoid presenting finite simulation endpoints as certainty.
            label = f'{p:.0%}'
            label = '<1%' if label == '0%' else '>99%' if label == '100%' else label
            probability = ('<div class="majority-probability">'
                           '<small>Estimated probability of a seat majority</small>'
                           f'<b>{escape(label)}</b><small>175 or more seats</small></div>')
        cards.append(f'<div class="bloc-card"><span>{escape(bloc["name"])}</span>'
                     f'<strong>{bloc["seats"]} <span style="font-size:1rem;font-weight:400">seats</span></strong>'
                     f'<small>{detail}</small>{ranges}{probability}</div>')
    allocation = state['seat_allocation']
    tie_note = (f'<p>{len(allocation["ties"])} exact quotient tie(s) were resolved by a reproducible forecast lottery; '
                'an official lottery could differ.</p>' if allocation['ties'] else '')
    probability_note = (
        f'<p>Majority probabilities are the fraction of {state["uncertainty"]["draws"]:,} simulated elections '
        'where each bloc reaches 175 seats. They depend on the model and its uncertainty assumptions; '
        'historical testing does not establish calibrated probabilities. '
        'Values rounding to 0% or 100% are shown as &lt;1% or &gt;99%.</p>' if probabilities else '')
    return ('<h2>Projected Riksdag seats</h2><p>349 seats · 175 needed for a majority. '
            'Headline numbers are the point projection.</p><div class="bloc-cards">' + ''.join(cards) + '</div>'
            + probability_note + '<p>The two groupings below are V + S + MP + C and M + L + KD + SD.</p>' + tie_note)


def uncertainty_html(state):
    uncertainty = state.get('uncertainty')
    if not uncertainty:
        return ''
    threshold = [p['party'] for p in uncertainty['parties'] if p['party'] != 'ÖVR'
                 and p['vote_share_pct']['lower_90'] <= 4 <= p['vote_share_pct']['upper_90']]
    threshold_note = ('<p>The 90% vote range crosses the 4% national threshold for '
                      + ', '.join(escape(p) for p in threshold)
                      + '. Its seat outcomes can form separate groups below and above the threshold.</p>' if threshold else '')
    return (f'<h2>Uncertainty in the final seats</h2><p>Based on {uncertainty["draws"]:,} joint simulations. '
            'Each simulated election allocates all 349 seats. The ranges include uncertain outstanding votes, '
            'shared reporting errors, late votes and an assumed allowance for final-count revisions.</p>'
            '<p class="test-note">These are approximate model-based ranges under stated assumptions. '
            'Historical checks cover one election and do not establish calibrated majority probabilities. '
            'A 90% range can miss systematic errors; it is not a guarantee.</p>' + threshold_note)


def plot_bloc_uncertainty(state):
    uncertainty = state.get('uncertainty')
    if not uncertainty:
        return None
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.1), sharex=True, layout='constrained')
    lower = min(b['seats']['lower_90'] for b in uncertainty['blocs'])
    upper = max(b['seats']['upper_90'] for b in uncertainty['blocs'])
    # Display the full simulated mass, including tails and threshold jumps.
    lower = min(lower, min(h['seats'] for b in uncertainty['blocs'] for h in b['seat_histogram']))
    upper = max(upper, max(h['seats'] for b in uncertainty['blocs'] for h in b['seat_histogram']))
    for ax, bloc, point, color in zip(axes, uncertainty['blocs'], state['blocs'], ['#b84349', '#3276ab']):
        hist = bloc['seat_histogram']
        ax.bar([h['seats'] for h in hist], [100 * h['count'] / uncertainty['draws'] for h in hist],
               width=.85, color=color, alpha=.8)
        interval = bloc['seats']
        ax.axvspan(interval['lower_90'] - .5, interval['upper_90'] + .5, color=color, alpha=.07)
        ax.axvspan(interval['lower_50'] - .5, interval['upper_50'] + .5, color=color, alpha=.11)
        ax.axvline(174.5, color='#263b50', linestyle='--', linewidth=1.2)
        ax.axvline(point['seats'], color=color, linestyle=':', linewidth=1.3)
        ax.set(title=bloc['name'], ylabel='Simulations (%)', xlim=(min(lower - 2, 172), max(upper + 2, 177)))
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].text(.99, .94, 'Dashed line: 175+ seats for a majority', transform=axes[0].transAxes,
                 ha='right', va='top', fontsize=9, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})
    axes[-1].set_xlabel('Final seats · shaded bands contain the central 50% and 90% of simulations')
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig


def plot_bloc_seats(state):
    if not state.get('seat_allocation'):
        return None
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    seats = state['seat_allocation']['national_seats']
    fig, ax = plt.subplots(figsize=(10, 3.6), layout='constrained')
    for row, bloc in enumerate(state['blocs']):
        left = 0
        for party in bloc['parties']:
            count = seats[party]
            ax.barh(row, count, left=left, color=PARTY_COLORS[party], height=.48,
                    edgecolor='white', linewidth=1)
            if count >= 10:
                ax.text(left + count / 2, row, f'{party}\n{count}', ha='center', va='center',
                        color='#182536' if party in ('SD', 'L') else 'white', fontsize=10, weight='bold')
            left += count
        ax.text(left + 3, row, str(left), ha='left', va='center', fontsize=13, weight='bold', color='#182536')
    ax.axvline(175, color='#263b50', linestyle='--', linewidth=1.3, zorder=0)
    ax.text(175, 1.02, '175 · majority', transform=ax.get_xaxis_transform(), ha='center', fontsize=10)
    ax.set(yticks=range(2), yticklabels=[b['name'] for b in state['blocs']],
           xlim=(0, max(200, max(b['seats'] for b in state['blocs']) + 30)), xlabel='Projected seats')
    ax.invert_yaxis()
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.legend(handles=[Patch(color=PARTY_COLORS[p], label=f'{p}: {seats[p]}')
                       for bloc in state['blocs'] for p in bloc['parties']],
              loc='upper center', bbox_to_anchor=(.5, -.23), ncol=8, frameon=False, fontsize=9)
    return fig


def plot_shares(state):
    if not state['parties'] or 'forecast_share_pct' not in state['parties'][0]:
        return None
    import matplotlib.pyplot as plt
    table = pd.DataFrame(state['parties'])
    fig, ax = plt.subplots(figsize=(10, 4.3), layout='constrained')
    x = np.arange(len(table))
    ax.bar(x - .18, table.raw_reported_share_pct, .35, color='#aebbc9', label='Counted share')
    ax.bar(x + .18, table.forecast_share_pct, .35, color='#203c5f', label='Estimated eventual share')
    if 'vote_share_lower_90' in table:
        ax.vlines(x + .18, table.vote_share_lower_90, table.vote_share_upper_90,
                  color='#a82332', linewidth=1.4, label='90% model range')
        ax.plot(x + .18, table.vote_share_lower_90, '_', color='#a82332')
        ax.plot(x + .18, table.vote_share_upper_90, '_', color='#a82332')
    ax.set(xticks=x, xticklabels=table.party, ylabel='Valid votes (%)', title='Counted votes and estimated eventual result')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False)
    return fig


def plot_coverage(state):
    if not state.get('coverage_by_county'):
        return None
    import matplotlib.pyplot as plt
    table = pd.DataFrame(state['coverage_by_county'])
    fig, ax = plt.subplots(figsize=(10, 3.2), layout='constrained')
    ax.bar(table.county, table.reported / table.total * 100, color='#557793')
    ax.set(ylim=(0,100), ylabel='Districts reported (%)', xlabel='County code', title='Geographical reporting coverage')
    ax.spines[['top','right']].set_visible(False)
    return fig


def plot_history(state):
    history = [h for h in state.get('history', []) if 'blocs' in h]
    if len(history) < 2:
        return None
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.3), layout='constrained')
    for bloc, color in zip(history[-1]['blocs'], ['#b84349', '#3276ab']):
        lo = [h.get('bloc_intervals', {}).get(bloc, {}).get('lower_90', np.nan) for h in history]
        hi = [h.get('bloc_intervals', {}).get(bloc, {}).get('upper_90', np.nan) for h in history]
        ax.fill_between(range(1, len(history) + 1), lo, hi, color=color, alpha=.10)
        ax.plot(range(1, len(history) + 1), [h['blocs'][bloc] for h in history],
                label=bloc, color=color, marker='o', markersize=3)
    ax.axhline(175, color='#263b50', linestyle='--', linewidth=1, label='175-seat majority')
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlabel='Distinct data updates · shading shows 90% model ranges where available',
           ylabel='Projected seats', title='How the bloc seat estimates have changed')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(ncol=3, frameon=False, fontsize=9)
    return fig


def methods_html(state):
    return '''<h2>How to read this report</h2><p>The estimate combines reported votes with predictions for districts still outstanding, including late-vote pools. It uses final 2022 district results mapped to the 2026 geography and the 2026 electorate. A regularized regression learns local changes from the ordinary districts that have reported; completed district counts are preserved exactly.</p>
<p>Forecast votes are added up in each of the 29 Riksdag constituencies, including their late-vote pools. The seat calculation uses the official 2026 distribution of 310 fixed seats and 39 adjustment seats, the modified Sainte-Laguë method, the 4% national threshold and the 12% constituency exception. Excess fixed seats are returned and redistributed under the electoral rules before adjustment seats are placed.</p>
<p>The pooled ÖVR category contributes to the threshold denominator but receives no seats. The projection assumes no individual party within that pool qualifies nationally or locally. It allocates seats between parties; candidate shortages are not modelled.</p>
<p>The headline seats are implied by the unchanged point vote forecast. The accompanying ranges come from approximate Bayesian predictive simulations: shared regression-parameter draws and district variation, with explicit additional assumptions about reporting imbalance, late votes and final-count revisions. Every draw is allocated separately. The simulated mean or median may differ from the point projection; individual party medians and interval endpoints need not add to 349.</p>
<p>Reported votes remain fixed within simulations of the preliminary count. A separate assumed revision effect produces possible final counts, so the final-result ranges need not collapse when every preliminary district has reported. This does not download or mix in the final feed. The scale of systematic errors is partly assumed, and the historical checks reuse one election; displayed ranges are not calibrated guarantees. Small vote changes around a threshold can move many seats. The two bloc totals represent the stated party groupings, not government formation. Majority probabilities count the joint simulations with at least 175 bloc seats. They are conditional on these assumptions and are not calibrated guarantees.</p>
<p>Unreported districts are missing observations, not zero votes. Revisions replace prior counts. A source older than 15 minutes is flagged even if the fetch succeeds; during the scheduled pause this is expected. Reload this static page to see a newly published report.</p>
<p>Sources: <a href="https://www.val.se/valresultat-och-statistik/statistik-och-data/teknisk-beskrivning-av-resultatfiler">Valmyndigheten’s result-feed documentation</a>, <a href="https://www.val.se/valresultat-och-statistik/statistik-och-data/radata-val-2026">2026 geography, electorate and fixed seats</a>, <a href="https://www.riksdagen.se/sv/dokument-och-lagar/dokument/svensk-forfattningssamling/vallag-2005837_sfs-2005-837/">Vallagen, chapter 14</a>. All displayed times are Swedish local time.</p>'''
