# %%
"""Refresh, validate and freeze the official background for the 2026 forecast.

Run before election night: uv run refresh-inputs.py
For a reproducible rebuild using existing downloads: add --offline.
No rehearsal votes or metadata enter these inputs. The immutable snapshot and
its manifest are committed before a single atomic active-pointer replacement.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
from urllib.parse import urljoin, unquote

import numpy as np
import pandas as pd
from val2026.election_data import PARTIES, NAMES, build_baseline, mapping_2026
from val2026.election_feed import HttpClient, atomic_write, immutable_write, json_bytes

ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
RAW_2026 = 'https://www.val.se/valresultat-och-statistik/statistik-och-data/radata-val-2026'
RAW_2022 = 'https://www.val.se/valresultat-och-statistik/statistik-och-data/radata-fran-val-2002-2022'
SOURCES = {
    'riksdag-2022-final.xlsx': (RAW_2022, 'Roster-per-distrikt-slutligt-antal-roster-inklusive-totalt-valdeltagande-riksdagsvalet-2022.xlsx'),
    'valdistrikt-jamforelser-mellan-2022-och-2026.xlsx': (RAW_2026, 'valdistrikt-jamforelser-mellan-2022-och-2026.xlsx'),
    'electorate-2026.xlsx': (RAW_2026, 'antal-rostberattigade-per-valdistrikt-uppdelat-pa-kon-och-alder-kvalifikationsdagen-14-augusti-2026-val-till-riksdagen.xlsx'),
    'fasta-valkretsmandat-val-2026.xlsx': (RAW_2026, 'fasta-valkretsmandat-val-2026.xlsx'),
}

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(href)


def official_urls(client):
    pages = {}
    result = {}
    for filename, (page, ending) in SOURCES.items():
        if page not in pages:
            parser = Links()
            parser.feed(client.get(page, 5_000_000).decode())
            pages[page] = parser.links
        candidates = {urljoin(page, href) for href in pages[page]
                      if unquote(href).lower().endswith(ending.lower())}
        if len(candidates) != 1:
            raise ValueError(f'Expected one current official link for {filename}, found {len(candidates)}; inspect {page}')
        result[filename] = candidates.pop()
    return result


# %% Read final 2022 counts without depending on preliminary archives/backtests.
def final_2022(path):
    raw = pd.read_excel(path, sheet_name='roster_RD')
    raw.columns = raw.columns.str.strip()
    raw['Parti'] = raw.Parti.str.strip()
    meta = raw.drop_duplicates('Distrikt').copy()
    meta['kind'] = np.where(meta.Valdistriktnamn.str.contains('Uppsamlings', case=False), 'late', 'ordinary')
    meta['code'] = [str(c).zfill(6 if k == 'late' else 8) for c, k in zip(meta.Valdistriktskod, meta.kind)]
    meta['municipality'] = meta.code.str[:4]
    meta['county'] = meta.code.str[:2]
    meta = meta.rename(columns={'Röstberättigade': 'electorate', 'Valdistriktnamn': 'name'})
    meta = meta.set_index('Distrikt')[['code', 'municipality', 'county', 'kind', 'electorate', 'name']]
    named = raw[raw.Parti.isin(NAMES)].assign(party=lambda d: d.Parti.map(NAMES))
    votes = named.pivot_table(index='Distrikt', columns='party', values='Röster', aggfunc='sum').reindex(columns=PARTIES[:-1]).fillna(0)
    valid = raw[raw.Parti.eq('Summa giltiga röster')].set_index('Distrikt')['Röster']
    votes['ÖVR'] = valid - votes.sum(axis=1)
    prior = meta.join(votes, validate='one_to_one').reset_index(drop=True)
    if prior[PARTIES].isna().any().any() or (prior[PARTIES] < 0).any().any() or not prior.code.is_unique:
        raise ValueError('Invalid or duplicated final 2022 counts')
    if int(prior[PARTIES].sum().sum()) != 6477970:
        raise ValueError('2022 valid votes differ from the validated official final total; review revision')
    return prior


def validate_constituencies(ordinary):
    """Validate actual Riksdag geography, which differs from the 21 counties."""
    codes = ordinary.constituency
    names = ordinary.constituency_name
    if codes.isna().any() or not codes.str.fullmatch(r'\d{2}').all():
        raise ValueError('Missing or malformed Riksdag constituency identifiers')
    if set(codes) != {f'{i:02d}' for i in range(1, 30)}:
        raise ValueError('Expected all 29 official Riksdag constituencies')
    if names.isna().any() or names.str.strip().eq('').any():
        raise ValueError('Missing Riksdag constituency names')
    if (ordinary.groupby('constituency').constituency_name.nunique() != 1).any():
        raise ValueError('A Riksdag constituency has inconsistent names')
    if (ordinary.groupby('constituency_name').constituency.nunique() != 1).any():
        raise ValueError('A Riksdag constituency name has multiple identifiers')
    if (ordinary.groupby('municipality').constituency.nunique() != 1).any():
        raise ValueError('Municipality late-vote pool cannot map to one Riksdag constituency')


def fixed_seats_2026(path, ordinary):
    """Join the official seat decision to official identifiers by unique names.

    The seat workbook's Län column is a county code, not a Riksdag
    constituency code. The electorate workbook supplies the latter.
    """
    raw = pd.read_excel(path, sheet_name='Antal fasta valkretsmandat')
    raw.columns = raw.columns.str.strip()
    selected = raw.loc[raw.Valtyp.str.strip().eq('Val till Riksdagen'),
                       ['Valkrets', 'Fasta mandat', 'Totalt antal mandat']].copy()
    selected = selected.rename(columns={'Valkrets': 'name', 'Fasta mandat': 'fixed_seats'})
    selected['name'] = selected.name.str.strip()
    lookup = ordinary[['constituency', 'constituency_name']].drop_duplicates().rename(
        columns={'constituency_name': 'name'})
    if len(selected) != 29 or not selected.name.is_unique or set(selected.name) != set(lookup.name):
        raise ValueError('Fixed-seat decision and electorate constituency rosters differ')
    if not selected['Totalt antal mandat'].eq(349).all():
        raise ValueError('Riksdag seat decision does not specify 349 total seats')
    seats = pd.to_numeric(selected.fixed_seats, errors='raise')
    if not np.isfinite(seats).all() or not seats.gt(0).all() or (seats % 1).any() or seats.sum() != 310:
        raise ValueError('Riksdag fixed seats must be positive integers summing to 310')
    selected['fixed_seats'] = seats.astype(int)
    return lookup.merge(selected[['name', 'fixed_seats']], on='name', validate='one_to_one')[
        ['constituency', 'name', 'fixed_seats']].sort_values('constituency').reset_index(drop=True)


def prepare(folder):
    prior = final_2022(folder / 'riksdag-2022-final.xlsx')
    electorate = pd.read_excel(folder / 'electorate-2026.xlsx', sheet_name='rostber_per_distrikt',
                               dtype={'Valdistriktskod': str, 'Kommunkod': str, 'Riksdagsvalkretskod': str})
    electorate = electorate.rename(columns={'Valdistriktskod': 'code', 'Kommunkod': 'municipality', 'Valdistrikt': 'name',
                                           'Totalt': 'electorate', 'Riksdagsvalkretskod': 'constituency',
                                           'Riksdagsvalkrets': 'constituency_name'})
    total_rows = electorate.loc[electorate.code.isna(), 'electorate'].dropna()
    ordinary = electorate.dropna(subset=['code']).copy()
    if not ordinary.code.is_unique or not ordinary.code.str.fullmatch(r'\d{8}').all():
        raise ValueError('Electorate contains duplicate or malformed district identifiers')
    if not ordinary.municipality.eq(ordinary.code.str[:4]).all() or not ordinary.electorate.gt(0).all():
        raise ValueError('Invalid electorate municipality or population')
    if len(total_rows) != 1 or ordinary.electorate.sum() != total_rows.iloc[0]:
        raise ValueError('Electorate district sums disagree with the national total')
    ordinary['constituency_name'] = ordinary.constituency_name.str.strip()
    validate_constituencies(ordinary)
    fixed_seats = fixed_seats_2026(folder / 'fasta-valkretsmandat-val-2026.xlsx', ordinary)
    mapping = mapping_2026(folder / 'valdistrikt-jamforelser-mellan-2022-och-2026.xlsx')
    if not mapping.code.is_unique or set(mapping.code) != set(ordinary.code):
        raise ValueError('Official mapping and electorate district rosters differ')
    if not mapping.comparison.isin(['Ej jämförbart', 'Kan jämföras', 'Kan jämföras mot flera']).all():
        raise ValueError(f'Unknown comparison flags: {mapping.comparison.unique().tolist()}')
    ordinary['county'] = ordinary.code.str[:2]
    ordinary['kind'] = 'ordinary'
    ordinary = ordinary[['code', 'municipality', 'county', 'constituency', 'constituency_name', 'name', 'electorate', 'kind']]
    late = ordinary.groupby(['municipality', 'county', 'constituency', 'constituency_name'], as_index=False).electorate.sum()
    late['code'], late['kind'], late['name'] = late.municipality + 'LATE', 'late', 'Municipality late-vote pool'
    baseline = build_baseline(prior, pd.concat([ordinary, late], ignore_index=True), mapping)
    if not np.isfinite(baseline.expected_votes).all() or baseline.expected_votes.le(0).any():
        raise ValueError('Invalid expected vote volume')
    summary = {'ordinary_districts': len(ordinary), 'late_municipality_pools': len(late),
               'electorate': int(ordinary.electorate.sum()), 'final_2022_valid_votes': 6477970,
               'riksdag_constituencies': len(fixed_seats), 'fixed_seats': int(fixed_seats.fixed_seats.sum()),
               'baseline_sources': baseline.baseline_source.value_counts().to_dict(),
               'production_collection_roster': 'Bootstrap from a complete signature-verified production result file.'}
    return baseline, ordinary[['code', 'municipality', 'county', 'constituency', 'constituency_name', 'name']], fixed_seats, summary


def refresh(root=ROOT, *, offline=False):
    started = datetime.now(timezone.utc).isoformat()
    client = HttpClient()
    urls = {} if offline else official_urls(client)
    with tempfile.TemporaryDirectory(prefix='val2026-inputs-') as temporary:
        folder = Path(temporary)
        sources = []
        for filename in SOURCES:
            payload = (root / 'data' / filename).read_bytes() if offline else client.get(urls[filename], 60_000_000)
            (folder / filename).write_bytes(payload)
            sources.append({'filename': filename, 'url': urls.get(filename), 'source_page': SOURCES[filename][0],
                            'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload),
                            'retrieved_at_utc': None if offline else started, 'offline_rebuild': offline})
        baseline, roster, fixed_seats, validation = prepare(folder)
        baseline_bytes, roster_bytes = baseline.to_csv(index=False).encode(), roster.to_csv(index=False).encode()
        fixed_seats_bytes = fixed_seats.to_csv(index=False).encode()
        content_id = hashlib.sha256(json_bytes([s['sha256'] for s in sources]) + baseline_bytes + roster_bytes + fixed_seats_bytes).hexdigest()[:24]
        snapshot = root / 'data/inputs/snapshots' / content_id
        for source in sources:
            immutable_write(snapshot / source['filename'], (folder / source['filename']).read_bytes())
        immutable_write(snapshot / 'baseline-2026.csv', baseline_bytes)
        immutable_write(snapshot / 'ordinary-roster.csv', roster_bytes)
        immutable_write(snapshot / 'fixed-seats-2026.csv', fixed_seats_bytes)
        manifest = {'prepared_at_utc': started, 'snapshot_id': content_id, 'sources': sources, 'validation': validation,
                    'baseline_path': str((snapshot / 'baseline-2026.csv').relative_to(root)),
                    'baseline_sha256': hashlib.sha256(baseline_bytes).hexdigest(),
                    'ordinary_roster_path': str((snapshot / 'ordinary-roster.csv').relative_to(root)),
                    'ordinary_roster_sha256': hashlib.sha256(roster_bytes).hexdigest(),
                    'fixed_seats_path': str((snapshot / 'fixed-seats-2026.csv').relative_to(root)),
                    'fixed_seats_sha256': hashlib.sha256(fixed_seats_bytes).hexdigest()}
        immutable_write(snapshot / 'manifests' / (hashlib.sha256(json_bytes(manifest)).hexdigest() + '.json'),
                        json_bytes(manifest))
        # Compatibility exports are not read by production. Write these before
        # activation too, so any failure leaves the prior active pointer intact.
        atomic_write(root / 'data/processed/baseline-2026.csv', baseline_bytes)
        for source in sources:
            atomic_write(root / 'data' / source['filename'], (folder / source['filename']).read_bytes())
        atomic_write(root / 'data/inputs/latest-success.json', json_bytes(manifest))
    return manifest


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    manifest = refresh(offline=args.offline)
    print(json.dumps(manifest['validation'], indent=2))
    print('Frozen inputs:', manifest['snapshot_id'])
