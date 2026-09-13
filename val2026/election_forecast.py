# %%
"""Forecast from an already verified local 2026 preliminary Riksdag snapshot.

uv run scripts/forecast-latest.py
uv run scripts/forecast-latest.py --environment production

This script never downloads or polls. Run scripts/collect-results.py first. Rehearsal
counts are artificial test data and are useful only for integration checks.
The frozen selected-model.json controls the regression; the historical baseline
uses actual 2022 results and pre-election 2026 electorate. Late-vote reporting
is checked against the downloaded official 2026 district roster (metadata only).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd

from val2026.election_data import PARTIES
from val2026.election_feed import atomic_write, json_bytes, source_freshness, validate_result
from val2026.forecast_model import SwingRegression
from val2026.seats import allocate_riksdag
from val2026.forecast_uncertainty import DEFAULT_DRAWS

from val2026 import ROOT
PARTY_CODES = {"0001": "M", "0004": "C", "0003": "L", "0068": "KD",
               "0055": "MP", "0002": "S", "0005": "V", "0110": "SD"}
BLOCS = (("V + S + MP + C", ("V", "S", "MP", "C")),
         ("M + L + KD + SD", ("M", "L", "KD", "SD")))


# %% Input validation and exact district alignment.
def load_snapshot(root, environment):
    cache = root / "data" / "collector" / environment / "preliminary"
    receipt_path = cache / "latest-success.json"
    if not receipt_path.exists():
        raise ValueError(f"No verified {environment} preliminary snapshot. Run scripts/collect-results.py "
                         f"--environment {environment} --once first.")
    receipt = json.loads(receipt_path.read_bytes())
    if (receipt.get("status") != "ok" or receipt.get("environment") != environment
            or receipt.get("count") != "preliminary"
            or receipt.get("test") is not (environment == "rehearsal")
            or receipt.get("signature", {}).get("status") != "verified"):
        raise ValueError("Collector receipt is unverified or belongs to the wrong environment/count")
    archive_path, json_path = [root / receipt[key] for key in ("archive_local", "result_json_local")]
    for path in (archive_path, json_path):
        if not path.resolve().is_relative_to(cache.resolve()):
            raise ValueError("Snapshot paths must remain inside the selected collector environment")
    payload = archive_path.read_bytes()
    if (hashlib.md5(payload).hexdigest() != receipt["archive_md5"]
            or hashlib.sha256(payload).hexdigest() != receipt["archive_sha256"]):
        raise ValueError("Local archive changed since official signature verification")
    result_bytes = json_path.read_bytes()
    documents, member_names = {}, {}
    with ZipFile(io.BytesIO(payload)) as zipped:
        for kind in ('rostfordelning', 'mandatfordelning', 'summering'):
            names = [name for name in zipped.namelist() if kind in name and name.endswith('.json')]
            if len(names) != 1 or names[0] not in receipt['signature']['json_files_verified']:
                raise ValueError(f"Expected one signature-verified {kind} JSON in the same archive")
            member = zipped.read(names[0])
            if kind == 'rostfordelning' and member != result_bytes:
                raise ValueError("Local district JSON differs from the signature-verified archive")
            documents[kind] = json.loads(member)
            member_names[kind] = names[0]
    result = json.loads(result_bytes)
    validate_result(result, environment, "preliminary")
    features, _ = load_inputs(root)
    municipality_constituencies = features.groupby('municipality').constituency.first().to_dict()
    from val2026.reconciliation import reconcile_aggregates
    receipt['aggregate_reconciliation'] = reconcile_aggregates(
        result, documents['mandatfordelning'], documents['summering'],
        environment, municipality_constituencies)
    receipt['aggregate_reconciliation']['signed_archive_members'] = member_names
    return receipt, result



def reporting_counts(district):
    """Return NaN-equivalent None until both reporting time and counts exist.

    Numerical zeros without a reporting timestamp are placeholders, not returns.
    Explicit zero party votes in a reported district remain real zeros.
    """
    if not district.get("rapporteringsTid"):
        return None
    distribution = district.get("rostfordelning")
    valid = distribution.get("rosterPaverkaMandat") if isinstance(distribution, dict) else None
    if not isinstance(valid, dict) or valid.get("antalRoster") is None:
        return None
    listed = valid.get("partiRoster")
    other = valid.get("rosterOvrigaPartier")
    if not isinstance(listed, list) or not isinstance(other, dict) or other.get("antalRoster") is None:
        return None
    counts = {}
    for party in listed:
        known = PARTY_CODES.get(str(party.get("partikod", "")).zfill(4))
        if not known:
            raise ValueError(f"Unexpected report party code: {party.get('partikod')}")
        key = known
        if key in counts:
            raise ValueError(f"Duplicate party in district {district['valdistriktskod']}: {key}")
        counts[key] = party.get("antalRoster")
    if any(p not in counts or counts[p] is None for p in PARTIES[:-1]):
        raise ValueError(f"Reported district {district['valdistriktskod']} lacks explicit counts for a report party")
    unknown = [p for p in counts if p not in PARTIES[:-1]]
    if unknown:
        raise ValueError(f"Unexpected preliminary report parties {unknown}; review ÖVR grouping before fitting")
    numeric = [counts[p] for p in PARTIES[:-1]] + [other["antalRoster"], valid["antalRoster"]]
    if any(type(value) is not int for value in numeric):
        raise ValueError("Reported party counts and subtotal must be explicit integers")
    values = np.array(numeric[:-1], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError(f"Invalid party counts in district {district['valdistriktskod']}")
    if not np.isclose(values.sum(), valid["antalRoster"], rtol=0, atol=0):
        raise ValueError(f"Party counts do not reconcile in district {district['valdistriktskod']}")
    return values


def align_observed(features, districts, roster, *, late_roster_complete=True):
    if not features.code.is_unique:
        raise ValueError("Duplicate baseline district codes")
    physical = features.kind.eq("ordinary")
    late = features.kind.eq("late")
    if not (physical | late).all():
        raise ValueError("Unexpected baseline district kind")
    roster_by_code = {d["valdistriktskod"]: d for d in roster}
    if len(roster_by_code) != len(roster):
        raise ValueError("Duplicate official roster district codes")
    roster_ordinary = {d["valdistriktskod"] for d in roster if d["valdistriktstyp"] == "valdistrikt"}
    if roster_ordinary != set(features.loc[physical, "code"]):
        raise ValueError("Official ordinary district roster does not match historical baseline coverage")
    late_by_municipality = {}
    for district in roster:
        if district["valdistriktstyp"] == "uppsamlingsdistrikt":
            late_by_municipality.setdefault(district["kommunkod"], set()).add(district["valdistriktskod"])
        elif district["valdistriktstyp"] != "valdistrikt":
            raise ValueError(f"Unknown official district type: {district['valdistriktstyp']}")
    if late_roster_complete and {m + "LATE" for m in late_by_municipality} != set(features.loc[late, "code"]):
        raise ValueError("Official late-vote municipalities do not match historical baseline coverage")
    current = {d["valdistriktskod"]: d for d in districts}
    if len(current) != len(districts):
        raise ValueError("Duplicate current district codes")
    extra = set(current) - set(roster_by_code)
    if extra:
        raise ValueError(f"Snapshot has codes outside the official baseline roster: {sorted(extra)[:12]}")
    reported = {}
    for code, district in current.items():
        reference = roster_by_code[code]
        if any(district.get(key) != reference[key] for key in ("valdistriktstyp", "kommunkod")):
            raise ValueError(f"District type/municipality changed for {code}; update the baseline roster")
        values = reporting_counts(district)
        if values is not None:
            reported[code] = values
    observed = np.full((len(features), len(PARTIES)), np.nan)
    partial_late = np.zeros_like(observed)
    incomplete = []
    for row, district in enumerate(features.itertuples()):
        if district.kind == "ordinary":
            if district.code in reported:
                observed[row] = reported[district.code]
        else:
            expected = late_by_municipality.get(district.municipality, set())
            arrived = expected & set(reported)
            if late_roster_complete and expected and arrived == expected:
                observed[row] = np.sum([reported[code] for code in sorted(expected)], axis=0)
            else:
                partial_late[row] = np.sum([reported[code] for code in sorted(arrived)], axis=0) if arrived else 0
                incomplete.append({"code": district.code, "municipality": district.municipality,
                                   "expected_records": len(expected) if late_roster_complete else None, "present_records": len(expected & set(current)),
                                   "reported_records": len(arrived), "unreported_codes": sorted(expected - arrived),
                                   "reported_valid_votes": int(partial_late[row].sum()),
                                   "reported_party_votes": dict(zip(PARTIES, partial_late[row].astype(int).tolist()))})
    raw = np.sum(list(reported.values()), axis=0) if reported else np.zeros(len(PARTIES))
    if not np.array_equal(np.nansum(observed, axis=0) + partial_late.sum(axis=0), raw):
        raise ValueError("Alignment did not preserve every reported party vote")
    summary = {"baseline_rows": len(features), "baseline_ordinary_districts": int(physical.sum()),
               "baseline_late_municipalities": int(late.sum()), "official_source_districts": len(roster),
               "source_records_present": len(current), "source_records_reported": len(reported),
               "missing_ordinary_records": sorted(roster_ordinary - set(current)),
               "ordinary_districts_observed": int((np.isfinite(observed).all(axis=1) & physical).sum()),
               "complete_late_municipalities": int((np.isfinite(observed).all(axis=1) & late).sum()),
               "incomplete_late_municipalities": incomplete}
    return observed, partial_late, raw, summary


def preserve_partial_lower_bounds(predicted, observed, partial_late):
    known = np.isfinite(observed).all(axis=1)
    if not np.array_equal(predicted[known], observed[known]):
        raise ValueError("The model changed a completed district's actual reported votes")
    if not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError("The model returned negative or nonfinite votes")
    corrections = (~known) & (predicted < partial_late).any(axis=1)
    # A partial late municipality cannot end below votes already counted. This
    # operational lower bound is not a separate fitted late-vote model.
    predicted = np.maximum(predicted, partial_late)
    return predicted, corrections


def read_hashed(root, relative, expected_hash):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Input path escapes the project")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"Frozen input changed: {relative}")
    return path, payload


def load_inputs(root):
    manifest = json.loads((root / "data/inputs/latest-success.json").read_bytes())
    path, _ = read_hashed(root, manifest["baseline_path"], manifest["baseline_sha256"])
    features = pd.read_csv(path, dtype={"code": str, "county": str, "municipality": str, "constituency": str})
    roster_path, _ = read_hashed(root, manifest["ordinary_roster_path"], manifest["ordinary_roster_sha256"])
    ordinary_roster = pd.read_csv(roster_path, dtype={"code": str, "municipality": str, "constituency": str})
    geography = ["code", "municipality", "constituency"]
    if (not ordinary_roster.code.is_unique or
            not ordinary_roster[geography].sort_values("code").reset_index(drop=True).equals(
                features.loc[features.kind.eq("ordinary"), geography].sort_values("code").reset_index(drop=True))):
        raise ValueError("Frozen baseline and official ordinary roster disagree")
    if not features.code.is_unique or features.empty:
        raise ValueError("Baseline is empty or has duplicate codes")
    prior = features[["prior_" + p for p in PARTIES]].to_numpy(float)
    if not np.isfinite(prior).all() or (prior < 0).any() or not np.allclose(prior.sum(axis=1), 1):
        raise ValueError("Historical party shares are invalid")
    if not np.isfinite(features.expected_votes).all() or (features.expected_votes <= 0).any():
        raise ValueError("Historical expected vote volumes are invalid")
    fixed = load_fixed_seats(root, manifest)
    if (features.constituency.isna().any() or set(features.constituency) != set(fixed)
            or features.groupby("municipality").constituency.nunique().ne(1).any()):
        raise ValueError("Baseline has missing or ambiguous constituency assignments")
    return features, manifest


def load_fixed_seats(root, manifest):
    path, _ = read_hashed(root, manifest["fixed_seats_path"], manifest["fixed_seats_sha256"])
    table = pd.read_csv(path, dtype={"constituency": str})
    values = table.fixed_seats.to_numpy(float)
    if (not table.constituency.is_unique or set(table.constituency) != {f"{c:02d}" for c in range(1, 30)}
            or not np.isfinite(values).all() or (values < 0).any()
            or (values != np.floor(values)).any() or values.sum() != 310):
        raise ValueError("Frozen 2026 fixed seats must cover 29 constituencies and total 310")
    return dict(zip(table.constituency, values.astype(int).tolist()))


def forecast_seats(features, predicted, fixed_seats):
    """Aggregate all ordinary and late forecasts before allocating any seats."""
    table = pd.DataFrame(predicted, columns=PARTIES)
    table["constituency"] = features.constituency.to_numpy()
    if table.constituency.isna().any():
        raise ValueError("Missing constituency in forecast")
    votes = table.groupby("constituency")[PARTIES].sum().to_dict(orient="index")
    return allocate_riksdag(votes, fixed_seats)


def bloc_seats(allocation):
    return [{"name": name, "parties": list(parties),
             "seats": sum(allocation["national_seats"].get(p, 0) for p in parties),
             "majority_threshold": 175}
            for name, parties in BLOCS]


def resolve_roster(root, environment, features, source, receipt):
    """Use official baseline geography; bootstrap late roster from a full signed feed.

    An incomplete first publication cannot prove how many late records exist.
    In that case all late pools remain incomplete, even when every currently
    visible late record has reported. No rehearsal file is read for production.
    """
    physical = [{"valdistriktskod": r.code, "kommunkod": r.municipality,
                 "valdistriktstyp": "valdistrikt"}
                for r in features.loc[features.kind.eq("ordinary")].itertuples()]
    physical_codes = {r["valdistriktskod"] for r in physical}
    municipalities = set(features.loc[features.kind.eq("late"), "municipality"])
    roster_path = root / "data/collector" / environment / "preliminary/roster.json"
    saved = json.loads(roster_path.read_bytes()) if roster_path.exists() else None
    records = source["valdistrikt"]
    membership = features.groupby("municipality").constituency.first().to_dict()
    for record in records:
        if record.get("kretskod") != membership.get(record.get("kommunkod")):
            raise ValueError(f"Feed constituency differs from the frozen geography for {record.get('valdistriktskod')}")
    keys = ("valdistriktskod", "kommunkod", "valdistriktstyp")
    current = [{key: d[key] for key in keys} for d in records]
    expected = source.get("antalValdistriktSomSkaRaknas")
    counted = source.get("antalValdistriktRaknade")
    if (type(expected) is not int or type(counted) is not int
            or not 0 <= counted <= len(current) <= expected):
        raise ValueError("Invalid or missing root expected/counted district totals")
    for d in current:
        if d["kommunkod"] not in municipalities:
            raise ValueError("Feed contains a municipality absent from the baseline")
        if d["valdistriktstyp"] not in ("valdistrikt", "uppsamlingsdistrikt"):
            raise ValueError("Feed contains an unknown district type")
        if d["valdistriktstyp"] == "valdistrikt" and d["valdistriktskod"] not in physical_codes:
            raise ValueError("Feed ordinary district is absent from the frozen baseline")
    complete = (type(expected) is int and expected == len(current)
                and {d["valdistriktskod"] for d in current if d["valdistriktstyp"] == "valdistrikt"} == physical_codes
                and {d["kommunkod"] for d in current if d["valdistriktstyp"] == "uppsamlingsdistrikt"} == municipalities)
    if saved:
        if saved.get("environment") != environment:
            raise ValueError("Saved roster has wrong environment")
        roster = saved["districts"]
        if complete and sorted(current, key=lambda d: d["valdistriktskod"]) != sorted(roster, key=lambda d: d["valdistriktskod"]):
            raise ValueError("Official roster changed; refresh and review geography before forecasting")
        if type(expected) is int and expected != len(roster):
            raise ValueError("Feed expected district count differs from the saved official roster")
        return roster, True, saved["provenance"]
    if complete:
        # Run alignment before persisting, to check municipalities and types too.
        align_observed(features, records, current)
        provenance = {"status": "validated_full_feed", "archive_sha256": receipt["archive_sha256"],
                      "source_updated_at": source.get("senasteUppdateringstid"),
                      "environment": environment, "district_count": len(current)}
        atomic_write(roster_path, json_bytes({"environment": environment, "districts": current,
                                            "provenance": provenance}))
        return current, True, provenance
    roster = physical + [d for d in current if d["valdistriktstyp"] == "uppsamlingsdistrikt"]
    return roster, False, {"status": "pending_full_feed", "environment": environment,
                          "explanation": "Ordinary geography verified against frozen inputs; all late pools remain incomplete."}


def frozen_model(root):
    lock = json.loads((root / "model-lock.json").read_bytes())
    for filename, expected_hash in lock["files"].items():
        read_hashed(root, filename, expected_hash)
    config = json.loads((root / "outputs/selected-model.json").read_bytes())
    name = config["selected_name"]
    return name, config["models"][name], lock


def evaluate_snapshot(root, environment, features, input_manifest, receipt, source, *, now=None, simulation_draws=DEFAULT_DRAWS):
    now = now or datetime.now(timezone.utc)
    roster, complete, provenance = resolve_roster(root, environment, features, source, receipt)
    observed, partial, raw, alignment = align_observed(features, source["valdistrikt"], roster,
                                                      late_roster_complete=complete)
    stated_counted = source.get("antalValdistriktRaknade")
    if type(stated_counted) is int and stated_counted != alignment["source_records_reported"]:
        raise ValueError("Feed counted-district total differs from district records with complete explicit party counts")
    name, options, lock = frozen_model(root)
    count = alignment["ordinary_districts_observed"]
    summary = {"generated_at_utc": now.isoformat(), "environment": environment,
               "test": environment == "rehearsal", "count": "preliminary", "model": name,
               "model_options": options, "model_lock": lock,
               "source_updated_at": source.get("senasteUppdateringstid"),
               "snapshot_checked_at_utc": receipt["checked_at_utc"],
               "archive_sha256": receipt["archive_sha256"], "archive_md5": receipt["archive_md5"],
               "archive_url": receipt["archive_url"], "roster_source": provenance,
               "aggregate_reconciliation": receipt.get("aggregate_reconciliation"),
               "baseline_sha256": input_manifest["baseline_sha256"],
               "fixed_seats_sha256": input_manifest["fixed_seats_sha256"],
               "raw_valid_votes": int(raw.sum()), **alignment,
               **source_freshness(source.get("senasteUppdateringstid"), now, 15)}
    known = np.isfinite(observed).all(axis=1)
    ordinary = features.kind.eq("ordinary").to_numpy()
    summary["counties_reporting"] = int(features.loc[known & ordinary, "county"].nunique())
    summary["counties_total"] = int(features.loc[ordinary, "county"].nunique())
    summary["expected_vote_weight_reported_pct"] = float(100 * features.loc[known & ordinary, "expected_votes"].sum()
                                                       / features.loc[ordinary, "expected_votes"].sum())
    summary["coverage_by_county"] = [
        {"county": c, "reported": int((known & mask).sum()), "total": int(mask.sum())}
        for c in sorted(features.county.unique())
        for mask in [features.county.eq(c).to_numpy() & ordinary]]
    raw_share = (raw / raw.sum() * 100).tolist() if raw.sum() else [None] * len(PARTIES)
    summary["parties"] = [{"party": p, "raw_reported_votes": int(raw[j]),
                           "raw_reported_share_pct": raw_share[j]} for j, p in enumerate(PARTIES)]
    if count < 2:
        summary.update(estimate_state="waiting", fit_predict_seconds=0, training_rows=0,
                       message="Waiting for at least two ordinary districts with complete results.")
        return summary, None
    model = SwingRegression(features, **options)
    predicted, details = model.fit_predict(observed, return_details=True)
    predicted, corrections = preserve_partial_lower_bounds(predicted, observed, partial)
    seat_started = perf_counter()
    fixed_seats = load_fixed_seats(root, input_manifest)
    allocation = forecast_seats(features, predicted, fixed_seats)
    summary.update(seat_allocation=allocation, blocs=bloc_seats(allocation),
                   seat_allocation_seconds=perf_counter() - seat_started)
    from val2026.forecast_uncertainty import forecast_uncertainty
    uncertainty = forecast_uncertainty(model, observed, predicted, partial, fixed_seats, draws=simulation_draws)
    summary['uncertainty'] = uncertainty
    uncertain_parties = {p['party']: p for p in uncertainty['parties']}
    for bloc, distribution in zip(summary['blocs'], uncertainty['blocs']):
        bloc['uncertainty'] = distribution['seats']
    totals = predicted.sum(axis=0)
    for j, item in enumerate(summary["parties"]):
        party = item["party"]
        item.update(forecast_votes=float(totals[j]), forecast_share_pct=float(totals[j] / totals.sum() * 100),
                    forecast_seats=allocation["national_seats"][party],
                    fixed_seats=allocation["national_fixed_seats"][party],
                    adjustment_seats=allocation["national_adjustment_seats"][party])
        item.update({f'seat_{key}': value for key, value in uncertain_parties[party]['seats'].items()})
        item.update({f'vote_share_{key}': value for key, value in uncertain_parties[party]['vote_share_pct'].items()})
    summary.update(estimate_state="early" if count < 300 else "forecast",
                   estimate_generated_at_utc=now.isoformat(), fit_predict_seconds=details["seconds"],
                   training_rows=details["training_rows"], regression_weight=details["regression_weight"],
                   forecast_valid_votes=float(totals.sum()),
                   partial_late_pool_lower_bounds_applied=features.loc[corrections, "code"].tolist(),
                   message="Early estimate: fewer than 300 districts; reporting is not representative."
                           if count < 300 else "Projected Riksdag seats from counted votes and estimates for districts still outstanding.")
    output = features[["code", "municipality", "county", "constituency", "constituency_name", "kind", "name"]].copy()
    output["observed_complete"] = known
    for j, party in enumerate(PARTIES):
        output["observed_" + party] = observed[:, j]
        output["partial_late_observed_" + party] = partial[:, j]
        output["forecast_" + party] = predicted[:, j]
    return summary, output
