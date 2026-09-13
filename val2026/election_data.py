"""Read official election files and construct historical-only district features."""
from pathlib import Path
import json
from zipfile import ZipFile
import numpy as np
import pandas as pd

from val2026 import ROOT
DATA = ROOT / 'data'
PARTIES = ['M', 'C', 'L', 'KD', 'MP', 'S', 'V', 'SD', 'ÖVR']
NAMES = {
    'Moderaterna': 'M', 'Centerpartiet': 'C', 'Liberalerna': 'L',
    'Liberalerna (tidigare Folkpartiet)': 'L', 'Kristdemokraterna': 'KD',
    'Miljöpartiet': 'MP', 'Miljöpartiet de gröna': 'MP',
    'Socialdemokraterna': 'S', 'Arbetarepartiet-Socialdemokraterna': 'S',
    'Vänsterpartiet': 'V', 'Sverigedemokraterna': 'SD',
}


def results_2022():
    cache = DATA / 'processed' / 'results-2022.csv'
    raw = pd.read_excel(DATA / 'riksdag-2022-final.xlsx', sheet_name='roster_RD')
    raw.columns = raw.columns.str.strip()
    raw['Parti'] = raw.Parti.str.strip()
    metadata = raw.drop_duplicates('Distrikt').copy()
    metadata['kind'] = np.where(metadata.Valdistriktnamn.str.contains('Uppsamlings', case=False), 'late', 'ordinary')
    metadata['code'] = [str(c).zfill(6 if k == 'late' else 8) for c, k in zip(metadata.Valdistriktskod, metadata.kind)]
    metadata['municipality'] = metadata.code.str[:4]
    metadata['county'] = metadata.code.str[:2]
    metadata = metadata.rename(columns={'Röstberättigade': 'electorate', 'Valdistriktnamn': 'name'})
    metadata = metadata.set_index('Distrikt')[['code', 'municipality', 'county', 'kind', 'electorate', 'name']]
    for phase, frame in [('final', raw), ('prelim', pd.read_excel(DATA / 'riksdag-2022-preliminary.xlsx'))]:
        frame.columns = frame.columns.str.strip()
        frame['Parti'] = frame.Parti.str.strip()
        valid = frame[frame.Parti.eq('Summa giltiga röster')].set_index('Distrikt')['Röster']
        named = frame[frame.Parti.isin(NAMES)].assign(party=lambda d: d.Parti.map(NAMES))
        votes = named.pivot_table(index='Distrikt', columns='party', values='Röster', aggfunc='sum').reindex(columns=PARTIES[:8]).fillna(0)
        # All other valid parties, excluding invalid/blank votes and total rows.
        if phase == 'prelim':
            # The Excel subtotal includes unregistered-party votes even though
            # the official JSON classifies them as invalid. Use explicit parties.
            votes['ÖVR'] = frame[frame.Parti.eq('Övriga anmälda partier')].set_index('Distrikt')['Röster']
            unregistered = frame[frame.Parti.eq('Ogiltiga röster - ej anmälda partier')].set_index('Distrikt')['Röster']
            valid = valid - unregistered
        else:
            votes['ÖVR'] = valid - votes.sum(axis=1)
        assert votes.notna().all().all() and (votes >= 0).all().all()
        assert votes.sum(axis=1).sum() == valid.sum()
        metadata = metadata.join(votes.add_prefix(phase + '_'), validate='one_to_one')
    assert metadata.filter(regex='^(final|prelim)_').notna().all().all()
    # The archived preliminary JSON corrects a swap of districts 01270007/8
    # in the Excel download. Use the archived counts with their report times.
    with ZipFile(DATA / '2022-feed/Val_20220911_preliminar_00_RD.zip') as z:
        filename = next(n for n in z.namelist() if 'rostfordelning' in n and n.endswith('.json'))
        archived = json.loads(z.read(filename))
    archive_rows = []
    for district in archived['valdistrikt']:
        valid = district['rostfordelning']['rosterPaverkaMandat']
        party_counts = {p['partiforkortning']: p['antalRoster'] for p in valid['partiRoster']}
        party_counts['ÖVR'] = valid['rosterOvrigaPartier']['antalRoster']
        archive_rows.append({'code': district['valdistriktskod'], **{'prelim_' + p: party_counts[p] for p in PARTIES}})
    archive_table = pd.DataFrame(archive_rows).set_index('code')
    for p in PARTIES:
        metadata['prelim_' + p] = metadata.code.map(archive_table['prelim_' + p])
    assert metadata.filter(like='prelim_').notna().all().all()
    cache.parent.mkdir(exist_ok=True, parents=True)
    metadata.reset_index().to_csv(cache, index=False)
    return metadata.reset_index()


def results_2018():
    d = pd.read_excel(ROOT / 'old/scb-riksdag-results-2018.xlsx', dtype={'Valdistriktskod': str})
    d = d.rename(columns={'Valdistriktskod': 'code', 'Valdistriktsnamn': 'name', 'Antal röstberättigade': 'electorate'})
    d['kind'] = np.where(d.name.str.contains('förtids', case=False), 'late', 'ordinary')
    d['municipality'] = d.code.str[:4]
    d['county'] = d.code.str[:2]
    assert d[PARTIES].sum().sum() == 6476725
    assert d.loc[d.kind.eq('ordinary'), 'code'].is_unique
    return d[['code', 'name', 'municipality', 'county', 'kind', 'electorate'] + PARTIES]


def mapping_2022():
    d = pd.read_excel(ROOT / 'old/jamforelser-2018-och-2022-valdistrikt-och-uppsamlingsdistrikt.xlsx', sheet_name='Fysiska valdistrikt', dtype=str)
    d = d.rename(columns={'Kod_2022': 'code', 'Jämförbart': 'comparison', 'Valdistriktskod2018': 'old1', 'Kod 2 2018': 'old2', 'Kod 3 2018': 'old3'})
    d['comparable'] = d.comparison.str.lower().ne('nej')
    return d[['code', 'comparison', 'comparable', 'old1', 'old2', 'old3']]


def mapping_2026(path=None):
    d = pd.read_excel(path or DATA / 'valdistrikt-jamforelser-mellan-2022-och-2026.xlsx', sheet_name='Jämförelser', dtype=str)
    d = d.rename(columns={'Valdistriktskod 2026': 'code', 'Jämförbarhet': 'comparison', 'Valdistriktskod 1 2022': 'old1', 'Valdistriktskod 2 2022': 'old2'})
    d['comparable'] = d.comparison.ne('Ej jämförbart')
    d['old3'] = None
    # Official row Tranås Norra-Sommen has '006870101'; its numerical code
    # is 06870101. Canonical eight-digit IDs avoid dropping that source silently.
    for column in ['old1', 'old2']:
        d[column] = d[column].map(lambda c: str(int(c)).zfill(8) if pd.notna(c) else None)
    return d[['code', 'comparison', 'comparable', 'old1', 'old2', 'old3']]


def build_baseline(prior, target, mapping):
    """Historical shares/rates. No target-election vote counts are accepted.

    Comparable districts use summed old counts and electorate. Unmatched ones
    use old municipality districts not consumed by comparable mappings, with
    municipality/county fallbacks. Shared old codes supply shares/rates only;
    current electorate controls volume, so old votes are not copied wholesale.
    Late votes are aggregated to municipality to avoid nonunique historical IDs.
    """
    ordinary = prior[prior.kind.eq('ordinary')].set_index('code')
    matched = mapping[mapping.comparable].melt(id_vars='code', value_vars=['old1', 'old2', 'old3']).dropna(subset=['value'])
    used = set(matched.value)
    residual = ordinary.loc[~ordinary.index.isin(used)]
    columns = PARTIES + ['electorate']
    municipal = ordinary.groupby('municipality')[columns].sum()
    county = ordinary.groupby('county')[columns].sum()
    residual_muni = residual.groupby('municipality')[columns].sum()
    late_muni = prior[prior.kind.eq('late')].groupby('municipality')[PARTIES].sum()
    lookups = mapping.set_index('code')
    rows = []
    for _, district in target.iterrows():
        code, muni, region = district['code'], district.municipality, district.county
        sources = []
        if district.kind == 'late':
            counts = late_muni.loc[muni, PARTIES].to_numpy(float) if muni in late_muni.index else np.zeros(len(PARTIES))
            prior_electorate = municipal.loc[muni, 'electorate'] if muni in municipal.index else 0
            # Late target electorate is the municipality's pre-election electorate.
            expected = counts.sum() * district.electorate / prior_electorate if prior_electorate else 0
            if counts.sum() == 0:
                counts = county.loc[region, PARTIES].to_numpy(float)
            origin = 'municipality_late'
            rate = expected / district.electorate if district.electorate else 0
        else:
            m = lookups.loc[code]
            if m.comparable:
                sources = [m[c] for c in ['old1', 'old2', 'old3'] if pd.notna(m[c])]
                missing = [c for c in sources if c not in ordinary.index]
                if missing:
                    raise ValueError(f'Mapped historical districts missing for {code}: {missing}')
            if sources:
                aggregate = ordinary.loc[sources, columns].sum()
                origin = 'district_mapping'
            elif muni in residual_muni.index and residual_muni.loc[muni, PARTIES].sum() > 0:
                aggregate = residual_muni.loc[muni]
                origin = 'municipality_residual'
            elif muni in municipal.index:
                aggregate = municipal.loc[muni]
                origin = 'municipality_average'
            else:
                aggregate = county.loc[region]
                origin = 'county_average'
            counts = aggregate[PARTIES].to_numpy(float)
            rate = counts.sum() / aggregate.electorate
            expected = district.electorate * rate
        shares = (counts + 0.5) / (counts.sum() + 0.5 * len(PARTIES))
        rows.append({**district.to_dict(), 'baseline_source': origin, 'prior_rate': rate,
                     'expected_votes': expected, 'old_codes': '|'.join(sources),
                     **{'prior_' + p: s for p, s in zip(PARTIES, shares)}})
    result = pd.DataFrame(rows)
    assert result.code.is_unique
    assert result.filter(like='prior_').notna().all().all()
    assert (result.expected_votes >= 0).all()
    return result


def target_2022(d):
    ordinary = d[d.kind.eq('ordinary')].copy()
    late = d[d.kind.eq('late')].groupby(['municipality', 'county'], as_index=False)[[phase + '_' + p for phase in ['final', 'prelim'] for p in PARTIES]].sum()
    late['code'] = late.municipality + 'LATE'
    late['kind'] = 'late'
    late['name'] = 'Municipality late-vote pool'
    late['electorate'] = late.municipality.map(ordinary.groupby('municipality').electorate.sum())
    return pd.concat([ordinary, late], ignore_index=True)
