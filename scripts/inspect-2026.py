# %%
"""Quick look at 2026 district boundaries and Riksdag REHEARSAL results.

Run: uv run scripts/inspect-2026.py
Source: Valmyndigheten. Downloaded 2026-09-05; see data/SOURCES.md.
All result figures in this snapshot are TEST DATA, not election results.
The script reads local files only and can also be run cell by cell.
"""
from pathlib import Path
import hashlib
import json
from zipfile import ZipFile

import pandas as pd

from val2026 import ROOT
DATA = ROOT / "data"
pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 140)

# %% Read the official district comparison workbook; preserve leading zeros.
workbook = DATA / "valdistrikt-jamforelser-mellan-2022-och-2026.xlsx"
information = pd.read_excel(workbook, sheet_name="Information", header=None)
district_mapping = pd.read_excel(workbook, sheet_name="Jämförelser", dtype="string")
district_mapping = district_mapping.rename(columns={
    "Valdistriktskod 2026": "valdistriktskod",
    "Valdistriktsnamn 2026": "valdistriktsnamn",
    "Kommun": "kommun",
    "Län": "lan",
    "Jämförbarhet": "jamforbarhet",
    "Valdistriktskod 1 2022": "kod_2022_1",
    "Valdistriktskod 2 2022": "kod_2022_2",
})
print("DISTRICT COMPARISON (actual 2026 geography)")
print(information.iloc[0, 0])
print(f"Rows: {len(district_mapping):,}; columns: {list(district_mapping.columns)}")
print(district_mapping.head().to_string(index=False))
print("\nComparability:")
print(district_mapping["jamforbarhet"].value_counts(dropna=False).to_string())
print("\nMissing values:")
print(district_mapping.isna().sum().to_string())
assert district_mapping["valdistriktskod"].notna().all()
assert district_mapping["valdistriktskod"].is_unique

# %% A long crosswalk, allowing two old districts to map to one new district.
crosswalk = district_mapping.melt(
    id_vars=["valdistriktskod", "jamforbarhet"],
    value_vars=["kod_2022_1", "kod_2022_2"],
    var_name="mapping_slot", value_name="valdistriktskod_2022",
).dropna(subset=["valdistriktskod_2022"])
print("\nCrosswalk sample:")
print(crosswalk.head().to_string(index=False))

# %% Read the saved archive directly, checking against the downloaded index.
# MD5 here checks consistency with the index; it is not signature verification.
archive = DATA / "Genrep_2026_preliminar_00_RD.zip"
index_entries = dict(
    (path, checksum)
    for checksum, path in (
        line.split() for line in (DATA / "genrep2026-index.md5").read_text().splitlines()
        if line.strip()
    )
)
expected_md5 = index_entries["./p/rd/" + archive.name]
assert hashlib.md5(archive.read_bytes()).hexdigest() == expected_md5
with ZipFile(archive) as z:
    result_file = next(n for n in z.namelist() if "rostfordelning" in n and n.endswith(".json"))
    results = json.loads(z.read(result_file))

assert results.get("test") is True, "This inspection expects rehearsal data."
print("\nRIKSDAG REHEARSAL — TEST DATA ONLY")
metadata = {k: v for k, v in results.items() if k != "valdistrikt"}
print(json.dumps(metadata, ensure_ascii=False, indent=2))

# %% One row per district, keeping reporting status and late-vote districts.
districts = pd.json_normalize([
    {k: v for k, v in district.items() if k != "rostfordelning"}
    for district in results["valdistrikt"]
])
districts["reported"] = districts["rapporteringsTid"].notna() & districts["rapporteringsTid"].ne("")
assert districts["valdistriktskod"].is_unique
print("\nDistrict types and reporting status:")
print(districts.groupby(["valdistriktstyp", "reported"]).size().to_string())
print("\nSample district metadata:")
print(districts[["valdistriktskod", "namn", "valdistriktstyp", "totaltAntalRoster", "rapporteringsTid"]].head().to_string(index=False))

# %% Flatten party votes, including the separate 'other parties' bucket.
# Preserve missing values; a district that has not reported is not a zero result.
rows = []
totals = []
for district in results["valdistrikt"]:
    distribution = district.get("rostfordelning") or {}
    valid = distribution.get("rosterPaverkaMandat") or {}
    invalid = distribution.get("rosterEjPaverkaMandat") or {}
    totals.append({
        "valdistriktskod": district["valdistriktskod"],
        "valid_votes": valid.get("antalRoster"),
        "invalid_votes": invalid.get("antalRoster"),
    })
    parties = list(valid.get("partiRoster") or [])
    if valid.get("rosterOvrigaPartier") is not None:
        parties.append({**valid["rosterOvrigaPartier"], "partiforkortning": "ÖVR", "partikod": None})
    for party in parties:
        rows.append({
            "valdistriktskod": district["valdistriktskod"],
            "party": party["partiforkortning"],
            "partikod": party.get("partikod"),
            "n_votes": party.get("antalRoster"),
            "n_votes_previous": party.get("antalRosterForegaendeVal"),
        })
votes = pd.DataFrame(rows).merge(
    districts[["valdistriktskod", "valdistriktstyp", "reported"]],
    validate="many_to_one",
)
district_totals = districts.merge(pd.DataFrame(totals), validate="one_to_one")
print(f"\nDistrict × party rows: {len(votes):,}")
print(votes.head(9).to_string(index=False))

# %% Reconcile the snapshot, then summarize TEST votes by district type.
party_sums = votes.groupby("valdistriktskod")["n_votes"].sum(min_count=1)
checks = district_totals.set_index("valdistriktskod").join(party_sums.rename("party_sum"))
reported = checks.loc[checks["reported"]]
assert (reported["party_sum"] == reported["valid_votes"]).all()
assert (reported["valid_votes"] + reported["invalid_votes"] == reported["totaltAntalRoster"]).all()
party_summary = votes.loc[votes["reported"]].groupby(["valdistriktstyp", "party"])["n_votes"].sum(min_count=1).unstack("valdistriktstyp")
print("\nTEST vote totals by party and district type:")
print(party_summary.to_string())
print("\nParty and district vote totals reconcile.")

# %% Check the join to actual 2026 geography; collection districts stay separate.
coverage = districts.merge(district_mapping, on="valdistriktskod", how="outer", validate="one_to_one", indicator=True)
print("\nRehearsal-to-workbook coverage:")
print(coverage.groupby(["_merge", "valdistriktstyp"], observed=True, dropna=False).size().to_string())
print("\nUnmatched sample:")
print(coverage.loc[coverage["_merge"] != "both", ["valdistriktskod", "namn", "valdistriktstyp", "_merge"]].head().to_string(index=False))
# Available for further work: district_mapping, crosswalk, districts, votes,
# district_totals, party_summary, coverage. Nothing is written by this script.
