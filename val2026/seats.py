"""Party-level Riksdag seats under the rules applicable since 2018.

Independent implementation of Regeringsformen 3:6–8 and Vallagen 14:2–5:
https://www.riksdagen.se/sv/dokument-och-lagar/dokument/svensk-forfattningssamling/vallag-2005837_sfs-2005-837/
Worked sequence: Valmyndigheten, Manual för mandatfördelning (2025), pp. 7–16:
https://www.val.se/download/18.162047b519a91d05331183a9/1761747515752/manual-mandatfordelning-val-v785-05.pdf

This forecasts seats between parties, not elected candidates. Candidate shortages
and personal votes are outside its scope. An aggregated other-parties category
must be excluded: its votes remain in the threshold denominator, but a collective
category cannot qualify as an individual party. Fractions retain the supplied
vote weights without rounding them into fictitious integer election results.
"""
from __future__ import annotations

from collections.abc import Mapping, Set
from decimal import Decimal
from fractions import Fraction
import math
from numbers import Integral, Real
import random


class AllocationError(ValueError):
    """The supplied votes and seat counts cannot yield a complete allocation."""


def _weight(value, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal, Fraction)):
        raise AllocationError(f'{label} must be a finite nonnegative number')
    if not math.isfinite(value) or value < 0:
        raise AllocationError(f'{label} must be a finite nonnegative number')
    # Decimal text makes e.g. 4.0 / 100.0 exactly the 4% boundary, and does not
    # manufacture decimal-place rounding or nearly-equal quotient lotteries.
    return Fraction(str(value))


def _seat_count(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise AllocationError(f'{label} must be a nonnegative integer')
    return int(value)


def _divisor(already: int, first: Fraction = Fraction(6, 5)) -> Fraction:
    return first if already == 0 else Fraction(2 * already + 1)


class _Lottery:
    def __init__(self, seed: int):
        self.random = random.Random(seed)
        self.events: list[dict] = []

    @staticmethod
    def label(item):
        if isinstance(item, tuple):
            return {'constituency': item[0], 'party': item[1]}
        return item

    def choose(self, quotients: dict, *, stage: str, minimum=False):
        if not quotients:
            raise AllocationError(f'No eligible recipient in {stage}')
        best = (min if minimum else max)(quotients.values())
        choices = sorted(k for k, value in quotients.items() if value == best)
        chosen = choices[0] if len(choices) == 1 else self.random.choice(choices)
        if len(choices) > 1:
            self.events.append({
                'stage': stage,
                'quotient': float(best),
                'candidates': [self.label(k) for k in choices],
                'chosen': self.label(chosen),
            })
        return chosen


def _apportion(votes: dict[str, Fraction], count: int, lottery: _Lottery,
               stage: str) -> dict[str, int]:
    seats = dict.fromkeys(votes, 0)
    for _ in range(count):
        quotients = {p: n / _divisor(seats[p]) for p, n in votes.items()}
        party = lottery.choose(quotients, stage=stage)
        seats[party] += 1
    return seats


def allocate_riksdag(
    votes_by_constituency: Mapping[str, Mapping[str, float]],
    fixed_seats: Mapping[str, int],
    *,
    total_seats: int = 349,
    excluded_parties: Set[str] = frozenset({'ÖVR'}),
    tie_seed: int = 0,
) -> dict:
    """Return a JSON-safe complete party and constituency seat allocation.

    Supply all 29 constituencies and their official fixed seats (310 total) for
    the real Riksdag; smaller counts are supported for transparent test cases.
    Every vote must be a finite nonnegative weight. Missing party keys within a
    constituency mean zero; missing constituencies are rejected. All supplied
    votes must be valid party votes, including the excluded ÖVR aggregate.

    Exact quotient ties use a reproducible seeded *forecast* lottery, recorded
    in ``ties``. This is not the official lottery conducted by Valmyndigheten.
    """
    total_seats = _seat_count(total_seats, 'total_seats')
    if total_seats == 0:
        raise AllocationError('total_seats must be positive')
    if isinstance(tie_seed, bool) or not isinstance(tie_seed, Integral):
        raise AllocationError('tie_seed must be an integer')
    if not votes_by_constituency or set(votes_by_constituency) != set(fixed_seats):
        raise AllocationError('Votes and fixed seats must cover exactly the same constituencies')
    if any(not isinstance(c, str) or not c for c in fixed_seats):
        raise AllocationError('Constituency identifiers must be nonempty strings')
    constituencies = sorted(fixed_seats)
    counts = {c: _seat_count(fixed_seats[c], f'fixed_seats[{c}]') for c in constituencies}
    fixed_total = sum(counts.values())
    if fixed_total > total_seats:
        raise AllocationError('Fixed seats exceed total seats')
    parties_set = set()
    for c, row in votes_by_constituency.items():
        if not isinstance(row, Mapping):
            raise AllocationError(f'Votes for {c} must be a party mapping')
        if any(not isinstance(p, str) or not p for p in row):
            raise AllocationError('Party identifiers must be nonempty strings')
        parties_set.update(row)
    parties = sorted(parties_set)
    votes = {c: {p: _weight(votes_by_constituency[c].get(p, 0), f'votes[{c}][{p}]')
                 for p in parties} for c in constituencies}
    totals = {p: sum((votes[c][p] for c in constituencies), Fraction()) for p in parties}
    local_totals = {c: sum(votes[c].values(), Fraction()) for c in constituencies}
    valid = sum(totals.values(), Fraction())
    if valid == 0 or any(local_totals[c] == 0 and counts[c] > 0 for c in constituencies):
        raise AllocationError('Positive valid votes are required in every constituency with fixed seats')
    national_eligible = {p for p in parties if p not in excluded_parties and totals[p] * 25 >= valid}
    local_only = {
        p: [c for c in constituencies if votes[c][p] > 0 and votes[c][p] * 25 >= 3 * local_totals[c]]
        for p in parties if p not in excluded_parties and p not in national_eligible
    }
    local_only = {p: codes for p, codes in local_only.items() if codes}
    lottery = _Lottery(int(tie_seed))
    eligible_here = {
        c: {p for p in parties if p in national_eligible or c in local_only.get(p, [])}
        for c in constituencies
    }
    fixed = {}
    for c in constituencies:
        apportioned = _apportion(
            {p: votes[c][p] for p in sorted(eligible_here[c])}, counts[c], lottery,
            f'fixed:{c}',
        )
        fixed[c] = {p: apportioned.get(p, 0) for p in parties}
    initial_fixed = {c: dict(row) for c, row in fixed.items()}
    national_fixed = {p: sum(fixed[c][p] for c in constituencies) for p in parties}

    # Local-only parties retain their fixed seats and never enter the national
    # allocation. Reserve these seats before apportioning the rest of Parliament.
    reserved = sum(national_fixed[p] for p in parties if p not in national_eligible)
    entitlement = _apportion(
        {p: totals[p] for p in sorted(national_eligible)}, total_seats - reserved,
        lottery, 'national_entitlement',
    )
    initial_entitlement = {p: entitlement.get(p, national_fixed[p]) for p in parties}

    returned: list[dict] = []
    vacancies = dict.fromkeys(constituencies, 0)
    protected_overhang: dict[str, int] = {}
    for p in sorted(national_eligible):
        while national_fixed[p] > entitlement[p]:
            # The quotient which won the last remaining fixed seat, not the
            # quotient for the next seat. Constituencies with 0–2 fixed seats
            # are protected from returns by Vallagen 14:4a.
            quotients = {
                (c, p): votes[c][p] / _divisor(fixed[c][p] - 1)
                for c in constituencies if counts[c] >= 3 and fixed[c][p] > 0
            }
            if not quotients:
                protected_overhang[p] = national_fixed[p] - entitlement[p]
                break
            c, _ = lottery.choose(quotients, stage=f'return:{p}', minimum=True)
            returned.append({'constituency': c, 'from_party': p,
                             'winning_quotient': float(quotients[(c, p)])})
            fixed[c][p] -= 1
            national_fixed[p] -= 1
            vacancies[c] += 1

    redistributed: list[dict] = []
    while sum(vacancies.values()):
        # Section 4b compares the next eligible quotient across *all* open
        # constituencies, updating the party's national cap after each seat.
        quotients = {
            (c, p): votes[c][p] / _divisor(fixed[c][p])
            for c in constituencies if vacancies[c] > 0
            for p in sorted(eligible_here[c] & national_eligible)
            if national_fixed[p] < entitlement[p]
        }
        c, p = lottery.choose(quotients, stage='redistribute_returned')
        fixed[c][p] += 1
        national_fixed[p] += 1
        vacancies[c] -= 1
        redistributed.append({'constituency': c, 'to_party': p,
                              'quotient': float(quotients[(c, p)])})

    # Usually the first total entitlement stands. With protected overhang,
    # remove fully represented parties and their fixed seats, as in section 5,
    # and recompute the remaining national total until no overhang remains.
    active = set(national_eligible)
    final_totals = {p: national_fixed[p] for p in parties if p not in active}
    targets = entitlement
    while active:
        satisfied = {p for p in active if national_fixed[p] >= targets[p]}
        if not satisfied:
            final_totals.update({p: targets[p] for p in active})
            break
        final_totals.update({p: national_fixed[p] for p in satisfied})
        active -= satisfied
        seats_left = total_seats - sum(final_totals.values())
        if seats_left < 0:
            raise AllocationError('Protected fixed seats exceed the available national total')
        targets = _apportion({p: totals[p] for p in sorted(active)}, seats_left,
                             lottery, 'national_adjustment_entitlement')
    if sum(final_totals.values()) != total_seats:
        raise AllocationError('No eligible parties remain for all adjustment seats')

    adjustment = {c: dict.fromkeys(parties, 0) for c in constituencies}
    for p in sorted(national_eligible):
        for _ in range(final_totals[p] - national_fixed[p]):
            quotients = {
                c: votes[c][p] / _divisor(fixed[c][p] + adjustment[c][p], Fraction(1))
                for c in constituencies if votes[c][p] > 0
            }
            c = lottery.choose(quotients, stage=f'adjustment:{p}')
            adjustment[c][p] += 1
    rows = {
        c: {'fixed': fixed[c], 'adjustment': adjustment[c],
            'total': {p: fixed[c][p] + adjustment[c][p] for p in parties}}
        for c in constituencies
    }
    national_seats = {p: sum(rows[c]['total'][p] for c in constituencies) for p in parties}
    national_adjustment = {p: national_seats[p] - national_fixed[p] for p in parties}
    if (sum(national_seats.values()) != total_seats
            or sum(national_adjustment.values()) != total_seats - fixed_total
            or any(sum(fixed[c].values()) != counts[c] for c in constituencies)
            or national_seats != final_totals):
        raise AllocationError('Internal seat-allocation reconciliation failed')
    return {
        'method': 'modified_sainte_lague_1.2_riksdag_2018_onwards',
        'total_seats': total_seats,
        'fixed_seats_total': fixed_total,
        'adjustment_seats_total': total_seats - fixed_total,
        'national_seats': national_seats,
        'national_fixed_seats': national_fixed,
        'national_adjustment_seats': national_adjustment,
        'constituencies': rows,
        'national_vote_totals': {p: float(totals[p]) for p in parties},
        'national_vote_shares': {p: float(totals[p] / valid) for p in parties},
        'national_eligible': sorted(national_eligible),
        'local_only_eligible': local_only,
        'excluded_parties': sorted(set(excluded_parties) & set(parties)),
        'initial_fixed_seats': initial_fixed,
        'initial_national_entitlement': initial_entitlement,
        'returned_seats': returned,
        'redistributed_seats': redistributed,
        'protected_overhang': protected_overhang,
        'tie_policy': 'reproducible_seeded_forecast_lottery_not_official_lottery',
        'tie_seed': int(tie_seed),
        'ties': lottery.events,
    }
