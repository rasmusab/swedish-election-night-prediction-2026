"""Joint vote simulations converted to party and bloc seat distributions.

The headline point forecast is deliberately separate from marginal summaries.
All simulated elections pass through the same complete Riksdag allocator.
"""
from time import perf_counter

import numpy as np

from val2026.election_data import PARTIES
from val2026.seats import allocate_riksdag

DEFAULT_DRAWS = 1000
DEFAULT_SEED = 2026
BLOCS = (("V + S + MP + C", ("V", "S", "MP", "C")),
         ("M + L + KD + SD", ("M", "L", "KD", "SD")))


def interval_summary(values, *, discrete=False):
    values = np.asarray(values)
    q = np.quantile(values, [.05, .25, .5, .75, .95],
                    method='inverted_cdf' if discrete else 'linear')
    convert = int if discrete else float
    return {'mean': float(values.mean()), 'median': convert(q[2]),
            'lower_90': convert(q[0]), 'lower_50': convert(q[1]),
            'upper_50': convert(q[3]), 'upper_90': convert(q[4])}


def seat_histogram(values):
    seats, counts = np.unique(values, return_counts=True)
    return [{'seats': int(s), 'count': int(n)} for s, n in zip(seats, counts)]


def allocate_vote_draws(vote_draws, constituency_codes, fixed_seats, *, seed=DEFAULT_SEED):
    votes = np.asarray(vote_draws, dtype=float)
    if (votes.ndim != 3 or votes.shape[1:] != (len(constituency_codes), len(PARTIES))
            or votes.shape[0] < 2 or not np.isfinite(votes).all() or (votes < 0).any()
            or len(set(constituency_codes)) != len(constituency_codes)
            or set(constituency_codes) != set(fixed_seats)):
        raise ValueError('Joint vote simulations need complete, finite constituency-by-party counts')
    started = perf_counter()
    seats = np.empty((len(votes), len(PARTIES)), dtype=int)
    tie_draws = 0
    for index, draw in enumerate(votes):
        allocation = allocate_riksdag(
            {code: dict(zip(PARTIES, row.tolist())) for code, row in zip(constituency_codes, draw)},
            fixed_seats, tie_seed=seed + index)
        seats[index] = [allocation['national_seats'][p] for p in PARTIES]
        tie_draws += bool(allocation['ties'])
    if not np.all(seats.sum(axis=1) == 349):
        raise ValueError('Every simulated election must allocate exactly 349 seats')
    return seats, {'seat_allocation_seconds': perf_counter() - started,
                   'draws_with_quotient_ties': tie_draws}


def summarize_vote_draws(vote_draws, constituency_codes, fixed_seats, *, seed=DEFAULT_SEED):
    votes = np.asarray(vote_draws, dtype=float)
    seats, diagnostics = allocate_vote_draws(votes, constituency_codes, fixed_seats, seed=seed)
    national = votes.sum(axis=1)
    shares = national / national.sum(axis=1, keepdims=True)
    parties = []
    for j, party in enumerate(PARTIES):
        parties.append({'party': party, 'seats': interval_summary(seats[:, j], discrete=True),
                        'vote_share_pct': interval_summary(shares[:, j] * 100),
                        'seat_histogram': seat_histogram(seats[:, j]),
                        'national_threshold_probability': None if party == 'ÖVR' else float((shares[:, j] >= .04).mean()),
                        'any_seat_probability': float((seats[:, j] > 0).mean())})
    blocs = []
    for name, members in BLOCS:
        total = seats[:, [PARTIES.index(p) for p in members]].sum(axis=1)
        blocs.append({'name': name, 'parties': list(members), 'seats': interval_summary(total, discrete=True),
                      'seat_histogram': seat_histogram(total),
                      'majority_probability': float((total >= 175).mean())})
    if not np.all(seats[:, :-1].sum(axis=1) == 349):
        raise ValueError('The two modelled blocs must jointly cover every simulated seat')
    return {'draws': len(votes), 'seed': seed, 'parties': parties, 'blocs': blocs,
            'seat_draw_parties': list(PARTIES), 'seat_draws': seats.tolist(), 'diagnostics': diagnostics}


def forecast_uncertainty(model, observed, predicted, partial_late, fixed_seats,
                         *, draws=DEFAULT_DRAWS, seed=DEFAULT_SEED):
    from val2026.uncertainty import simulate_constituency_votes
    started = perf_counter()
    simulated = simulate_constituency_votes(model, observed, predicted=predicted,
                                            partial_late=partial_late, draws=draws, seed=seed)
    summary = summarize_vote_draws(simulated.vote_draws, simulated.constituency_codes, fixed_seats, seed=seed)
    summary.update(method='approximate_bayesian_predictive_simulation',
                   target='eventual_final_party_votes_and_seats',
                   calibration_status='assumption_based_single_election_checks',
                   interval_description='Central 50% and 90% predictive ranges under explicit model assumptions',
                   config=simulated.config, simulation_diagnostics=simulated.diagnostics,
                   runtime_seconds=perf_counter() - started)
    return summary
