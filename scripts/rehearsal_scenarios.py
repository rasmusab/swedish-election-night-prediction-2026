"""Offline integration fixtures. Synthetic archives only; never used for live collection.

The transport/signature verifier is intentionally simulated here. Real OpenSSL
verification is covered separately and by live rehearsal collection.
"""
from copy import deepcopy
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

from val2026.election_feed import BASE_URLS, CERTIFICATE_URL, Collector, FetchError
from val2026.election_forecast import ROOT, PARTY_CODES
from val2026.election_report import run_cycle

MEMBER = 'Genrep_2026_preliminar_rostfordelning_00_RD.json'
MANDATE_MEMBER = 'Genrep_2026_preliminar_mandatfordelning_00_RD.json'
SUMMARY_MEMBER = 'Genrep_2026_preliminar_summering_RD.json'
ARCHIVE = 'p/rd/Genrep_2026_preliminar_00_RD.zip'


def copy_inputs(root):
    manifest = json.loads((ROOT / 'data/inputs/latest-success.json').read_bytes())
    lock = json.loads((ROOT / 'model-lock.json').read_bytes())
    files = ['data/inputs/latest-success.json', manifest['baseline_path'], manifest['ordinary_roster_path'], manifest['fixed_seats_path'],
             'model-lock.json', *lock['files']]
    for name in files:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)


def template():
    with ZipFile(ROOT / 'data/Genrep_2026_preliminar_00_RD.zip') as zipped:
        data = json.loads(zipped.read(next(n for n in zipped.namelist() if 'rostfordelning' in n and n.endswith('.json'))))
    # Preserve actual rehearsal counts but discard unused comparisons to keep
    # the synthetic replay small. Every fixture remains explicitly test=True.
    for d in data['valdistrikt']:
        for key in list(d):
            if key not in ('valdistriktskod','valdistriktstyp','kommunkod','kretskod','rapporteringsTid','rostfordelning'):
                del d[key]
    return data


def partial(source, ordinary_count, late_count=0, *, step=0, omit_unreported=False):
    data = deepcopy(source)
    stamp = (datetime(2026,9,13,20,tzinfo=timezone.utc)+timedelta(minutes=step)).isoformat()
    data['senasteUppdateringstid'] = stamp
    data['antalValdistriktRaknade'] = 0
    seen = {'valdistrikt': 0, 'uppsamlingsdistrikt': 0}
    records = []
    for district in data['valdistrikt']:
        kind = district['valdistriktstyp']
        limit = ordinary_count if kind == 'valdistrikt' else late_count
        seen[kind] += 1
        if seen[kind] <= limit:
            district['rapporteringsTid'] = stamp
            data['antalValdistriktRaknade'] += 1
        else:
            district['rapporteringsTid'] = None
            district['rostfordelning'] = None
            if omit_unreported:
                continue
        records.append(district)
    data['valdistrikt'] = records
    return data


def aggregate_documents(source, roster):
    """Build synthetic official-shaped aggregates from this fixture's returns.

    The full hidden roster supplies expected counts even when the fixture omits
    unreported records. This helper deliberately does not validate malformed
    district inputs: that is the application's job in the negative tests.
    """
    expected, counted = Counter(), Counter()
    votes = defaultdict(Counter)
    for district in roster['valdistrikt']:
        expected.update(('national', 'C' + district['kretskod'], 'M' + district['kommunkod']))
    for district in source['valdistrikt']:
        if not district.get('rapporteringsTid') or not district.get('rostfordelning'):
            continue
        valid = district['rostfordelning'].get('rosterPaverkaMandat') or {}
        keys = ('national', 'C' + district['kretskod'], 'M' + district['kommunkod'])
        counted.update(keys)
        for key in keys:
            for party in valid.get('partiRoster') or []:
                value = party.get('antalRoster')
                if type(value) is int:
                    votes[key][str(party.get('partikod')).zfill(4)] += value
            value = (valid.get('rosterOvrigaPartier') or {}).get('antalRoster')
            if type(value) is int:
                votes[key]['other'] += value

    def area(key):
        values = votes[key]
        distribution = {'rosterPaverkaMandat': {
            'antalRoster': sum(values.values()),
            'partiRoster': [{'partikod': code, 'antalRoster': values[code]} for code in PARTY_CODES],
            'rosterOvrigaPartier': {'antalRoster': values['other']}}}
        return {'antalValdistriktRaknade': counted[key],
                'antalValdistriktSomSkaRaknas': expected[key],
                'rapporteringsTid': source.get('senasteUppdateringstid') if counted[key] else None,
                'rostfordelning': distribution if counted[key] else None}

    header = {key: deepcopy(value) for key, value in source.items()
              if key not in ('valdistrikt', 'antalValdistriktRaknade', 'antalValdistriktSomSkaRaknas')}
    national = {'kod': '00', **area('national'),
                'valkretsLista': [{'kod': key[1:], **area(key)}
                                 for key in sorted(expected) if key.startswith('C')]}
    municipalities = [{'kommunkod': key[1:], **area(key)}
                      for key in sorted(expected) if key.startswith('M')]
    return {'rostfordelning': source,
            'mandatfordelning': {**header, 'valomrade': national},
            'summering': {**header, 'kommuner': municipalities,
                         'antalValdistriktRaknade': counted['national'],
                         'antalValdistriktSomSkaRaknas': expected['national']}}


class FixtureClient:
    def __init__(self, roster):
        self.roster = roster

    def set(self, source, *, documents=None):
        self.error = None
        documents = documents if documents is not None else aggregate_documents(source, self.roster)
        stream = io.BytesIO()
        with ZipFile(stream, 'w', ZIP_DEFLATED) as zipped:
            for kind, name in [('rostfordelning', MEMBER), ('mandatfordelning', MANDATE_MEMBER), ('summering', SUMMARY_MEMBER)]:
                if kind not in documents:
                    continue
                zipped.writestr(name, json.dumps(documents[kind]).encode())
                zipped.writestr(name[:-5] + '_sign.sha256', b'FIXTURE-NOT-AN-OFFICIAL-SIGNATURE')
        self.payload = stream.getvalue()
        self.index = f'{hashlib.md5(self.payload).hexdigest()} ./{ARCHIVE}\n'.encode()
    def get(self, url, max_bytes):
        if self.error:
            raise self.error
        if url.endswith('index.md5'):
            return self.index
        if url == CERTIFICATE_URL:
            return b'fixture certificate'
        return self.payload


class Scenario:
    def __init__(self, root, *, draws=32):
        self.root = root
        self.draws = draws
        copy_inputs(root)
        self.source = template()
        self.client = FixtureClient(self.source)
        self.now = datetime(2026,9,13,20,1,tzinfo=timezone.utc)
        self.collector = Collector(root, environment='rehearsal', client=self.client,
            verifier=lambda members, certificate: {'status':'verified',
                'json_files_verified':[name for name in members if name.endswith('.json')], 'fixture': True},
            now=lambda: self.now)
    def run(self, data=None):
        if data is not None:
            self.client.set(data)
        state = run_cycle(self.root, 'rehearsal', collector=self.collector, now=self.now, simulation_draws=self.draws)
        self.now += timedelta(minutes=1)
        return state
