# 2022 archived district reporting times

Source: Valmyndigheten, downloaded 2026-09-06.

- Index: https://resultat.val.se/resultatfiler/val2022/index.md5
- Preliminary archive: https://resultat.val.se/resultatfiler/val2022/p/rd/Val_20220911_preliminar_00_RD.zip
- Final archive: https://resultat.val.se/resultatfiler/val2022/s/rd/Val_20220911_slutlig_00_RD.zip

The saved ZIPs were verified against the MD5 index. 2022-report-times.csv
contains every preliminary district's reported `rapporteringsTid`, unchanged
from the source, and code/kind. Source timestamps have no timezone suffix;
retain their original local-clock strings for ordering. There are 6,264 ordinary
and 314 collection districts, all with populated times and unique codes.

These are archived report timestamps, not a first-appearance log. The
preliminary archive was last updated 2022-09-22 at 14:07:27. Ordinary dates:
5,702 on September 11, 560 on September 12, and two on September 22
(01270008 and 01270007). Collections: 310 on September 14, four on September 15.
Later corrections can overwrite earlier reporting times and vote counts.
A replay ordered by these values is more grounded than random ordering,
but cannot reconstruct the exact election-night sequence of original reports.
Use timestamps only to set replay order, never as a predictor.

Party normalization: the preliminary workbook's `Summa giltiga röster`
includes 3,481 invalid unregistered-party votes. Eight named parties plus
explicit `Övriga anmälda partier` sum to 6,445,298, matching the JSON's
`rosterPaverkaMandat.antalRoster`. The final workbook and final archive
correctly sum to 6,477,970 valid votes; do not apply the preliminary
subtotal correction to final data.

Reconciliation: all 6,578 final district rows and all nine party categories
match the final workbook exactly. The final archive was updated
2022-10-20T16:34:13. Preliminary counts differ from the workbook only
for 01270007 and 01270008, with exactly offsetting differences in
all party totals; these are also the two September 22 timestamps.
Prefer preliminary archive counts when using these archived timestamps
so counts and metadata refer to the same snapshot.
