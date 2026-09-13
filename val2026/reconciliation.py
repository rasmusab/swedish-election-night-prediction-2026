"""Cross-check reported district votes against the same archive's aggregates.

The three JSON files are generated separately: their update times (and file
write counters) need not match. Exact vote and district totals must match before
we publish a new forecast. A transient inconsistent archive can be retried on
the next update while the report retains its previous successful estimate.
"""
from collections import Counter, defaultdict

import numpy as np

from val2026.election_data import PARTIES
from val2026.election_feed import parse_source_timestamp, validate_result


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"Aggregate reconciliation: {label} must be a nonnegative integer")
    return value


def _areas(records, key, expected_codes, label):
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError(f"Aggregate reconciliation: missing {label} list")
    codes = [r.get(key) for r in records]
    if any(not isinstance(c, str) for c in codes) or len(codes) != len(set(codes)):
        raise ValueError(f"Aggregate reconciliation: missing or duplicate {label} codes")
    if set(codes) != set(expected_codes):
        raise ValueError(f"Aggregate reconciliation: {label} coverage differs from frozen geography")
    return dict(zip(codes, records))


def _zero_placeholder(distribution, label, party_codes):
    """Allow null/partial zero placeholders only in a genuinely unreported area."""
    if distribution is None:
        return
    if not isinstance(distribution, dict):
        raise ValueError(f"Aggregate reconciliation: invalid unreported {label}")
    valid = distribution.get('rosterPaverkaMandat')
    if valid is None:
        return
    if not isinstance(valid, dict):
        raise ValueError(f"Aggregate reconciliation: invalid unreported votes for {label}")
    values = [valid.get('antalRoster')]
    listed = valid.get('partiRoster')
    if listed is not None:
        if not isinstance(listed, list) or any(not isinstance(p, dict) for p in listed):
            raise ValueError(f"Aggregate reconciliation: invalid unreported parties for {label}")
        codes = [str(p.get('partikod', '')).zfill(4) for p in listed]
        if len(codes) != len(set(codes)) or set(codes) - set(party_codes):
            raise ValueError(f"Aggregate reconciliation: unknown or duplicate unreported party in {label}")
        values.extend(p.get('antalRoster') for p in listed)
    other = valid.get('rosterOvrigaPartier')
    if other is not None:
        if not isinstance(other, dict):
            raise ValueError(f"Aggregate reconciliation: invalid unreported other votes for {label}")
        values.append(other.get('antalRoster'))
    if any(value is not None and (type(value) is not int or value != 0) for value in values):
        raise ValueError(f"Aggregate reconciliation: {label} has votes but no reported districts")


def reconcile_aggregates(source, mandate, summary, environment, municipality_constituencies):
    """Return an audit summary or raise before accepting an inconsistent forecast.

    Only reported counts are compared; no forecasts or historical values enter
    this check. When the district file omits unreported records, the aggregate
    expected counts must still sum to the same national roster size, while each
    local expected count must be at least the number of records actually seen.
    """
    # Import at call time so the existing strict district parser remains the
    # single definition of which raw counts the live model actually consumes.
    from val2026.election_forecast import PARTY_CODES, reporting_counts

    documents = {'districts': source, 'mandates': mandate, 'municipalities': summary}
    identity = ('valtillfalle', 'valklass', 'valtyp', 'valdatum', 'tidigareValdatum', 'rakningstillfalle')
    for name, document in documents.items():
        if not isinstance(document, dict):
            raise ValueError(f"Aggregate reconciliation: {name} JSON is not an object")
        # Reuse the environment, election and stage checks without pretending
        # the aggregate JSON itself contains a district list.
        validate_result({**document, 'valdistrikt': []}, environment, 'preliminary')
        if any(document.get(key) != source.get(key) for key in identity):
            raise ValueError(f"Aggregate reconciliation: {name} election metadata differs")
        parse_source_timestamp(document.get('senasteUppdateringstid'))

    national = mandate.get('valomrade')
    if not isinstance(national, dict) or national.get('kod') != '00':
        raise ValueError('Aggregate reconciliation: missing national area 00')
    constituencies = _areas(national.get('valkretsLista'), 'kod',
                            {f'{i:02d}' for i in range(1, 30)}, 'constituencies')
    municipalities = _areas(summary.get('kommuner'), 'kommunkod',
                            municipality_constituencies, 'municipalities')
    records = source.get('valdistrikt')
    if not isinstance(records, list):
        raise ValueError('Aggregate reconciliation: missing district list')
    expected = _integer(source.get('antalValdistriktSomSkaRaknas'), 'national expected district count')
    counted = _integer(source.get('antalValdistriktRaknade'), 'national reported district count')
    if not counted <= len(records) <= expected:
        raise ValueError('Aggregate reconciliation: invalid national district coverage')
    complete_roster = len(records) == expected
    votes = defaultdict(lambda: np.zeros(len(PARTIES), dtype=np.int64))
    reported, present = Counter(), Counter()
    seen = set()
    for district in records:
        code = district.get('valdistriktskod')
        municipality, constituency = district.get('kommunkod'), district.get('kretskod')
        if not isinstance(code, str) or not code or code in seen:
            raise ValueError('Aggregate reconciliation: missing or duplicate district code')
        seen.add(code)
        if municipality not in municipality_constituencies:
            raise ValueError('Aggregate reconciliation: district municipality is outside frozen geography')
        if municipality_constituencies[municipality] != constituency:
            raise ValueError(f'Feed constituency differs from frozen geography for {code}')
        keys = ('national', 'C' + constituency, 'M' + municipality)
        present.update(keys)
        values = reporting_counts(district)
        if values is not None:
            reported.update(keys)
            for key in keys:
                votes[key] += values.astype(np.int64)
    if reported['national'] != counted:
        raise ValueError('Aggregate reconciliation: counted districts differ from complete explicit district returns')

    def check_area(area, key, label):
        actual_count = _integer(area.get('antalValdistriktRaknade'), label + ' reported district count')
        area_expected = _integer(area.get('antalValdistriktSomSkaRaknas'), label + ' expected district count')
        if actual_count != reported[key]:
            raise ValueError(f'Aggregate reconciliation: {label} reported district count differs from district file; retry next update')
        if (actual_count > area_expected or present[key] > area_expected
                or (complete_roster and area_expected != present[key])):
            raise ValueError(f'Aggregate reconciliation: {label} expected district count differs from district file')
        if actual_count == 0:
            total_votes = area.get('totaltAntalRoster')
            if total_votes is not None and (type(total_votes) is not int or total_votes != 0):
                raise ValueError(f'Aggregate reconciliation: {label} has votes but no reported districts')
            _zero_placeholder(area.get('rostfordelning'), label, PARTY_CODES)
            return area_expected
        # Parse votes independently of aggregate reporting timestamps. They are
        # metadata for the last contributing district, not file snapshot IDs.
        wrapper = {**area, 'valdistriktskod': label, 'rapporteringsTid': 'reported aggregate'}
        values = reporting_counts(wrapper)
        if values is None:
            raise ValueError(f'Aggregate reconciliation: {label} lacks explicit reported party votes')
        if not np.array_equal(values, votes[key]):
            delta = {p: int(values[j] - votes[key][j]) for j, p in enumerate(PARTIES)
                     if values[j] != votes[key][j]}
            raise ValueError(f'Aggregate reconciliation: {label} party votes differ from district file {delta}; retry next update')
        return area_expected

    if check_area(national, 'national', 'national') != expected:
        raise ValueError('Aggregate reconciliation: national expected counts differ across files')
    if (_integer(summary.get('antalValdistriktRaknade'), 'summary reported district count') != counted
            or _integer(summary.get('antalValdistriktSomSkaRaknas'), 'summary expected district count') != expected):
        raise ValueError('Aggregate reconciliation: summary national counts differ from district file')
    for label, areas, prefix in [('constituency', constituencies, 'C'), ('municipality', municipalities, 'M')]:
        area_expected = sum(check_area(area, prefix + code, f'{label} {code}') for code, area in areas.items())
        if area_expected != expected:
            raise ValueError(f'Aggregate reconciliation: {label} expected counts do not sum to national count')
    return {'status': 'reconciled', 'national_areas_checked': 1,
            'constituencies_checked': len(constituencies), 'municipalities_checked': len(municipalities),
            'reported_districts': counted, 'expected_districts': expected,
            'district_roster_complete': complete_roster,
            'reported_valid_votes': int(votes['national'].sum()),
            'reported_party_votes': dict(zip(PARTIES, votes['national'].tolist())),
            'source_updated_at': {key: doc.get('senasteUppdateringstid') for key, doc in documents.items()},
            'source_update_counters': {key: doc.get('antalUppdateringar') for key, doc in documents.items()}}
