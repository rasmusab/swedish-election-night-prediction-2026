# Source downloads

Source: Valmyndigheten. Downloaded 2026-09-05.

- District comparison workbook (actual 2026 geography): https://www.val.se/download/18.1a2972da19f159e73fd3a47/1787064655347/valdistrikt-jamforelser-mellan-2022-och-2026.xlsx
- Rehearsal index: https://resultat.val.se/resultatfiler/genrep2026/index.md5
- Preliminary Riksdag rehearsal archive: https://resultat.val.se/resultatfiler/genrep2026/p/rd/Genrep_2026_preliminar_00_RD.zip
- JSON field documentation: https://www.val.se/download/18.1a2972da19f159e73fd324b/1786558916266/prel-rostfordelning.md

The archive is a rehearsal snapshot, explicitly marked `test: true`. Its counts
must not be presented as actual 2026 election results. It includes late-vote
collection districts as well as ordinary polling districts.

Production index for election night: https://resultat.val.se/resultatfiler/val2026/index.md5

The inspection script deliberately reads only the saved snapshot. It does not
poll, refresh, or silently switch to production. The MD5 check checks the saved
archive against the saved index, not the cryptographic signatures.

## Historical and electorate inputs added 2026-09-05

- 2022 final Riksdag district results: https://www.val.se/download/18.162047b519a91d0533118f4b/1764336897948/Roster-per-distrikt-slutligt-antal-roster-inklusive-totalt-valdeltagande-riksdagsvalet-2022.xlsx
- 2022 preliminary Riksdag district results: https://www.val.se/download/18.162047b519a91d05331197bf/1663915218435/preliminart-roster-per-distrikt-riksdagsvalet-2022.xlsx
- 2026 electorate on qualification day, 14 August: https://www.val.se/download/18.1a2972da19f159e73fd3b4a/1787064446298/antal-rostberattigade-per-valdistrikt-uppdelat-pa-kon-och-alder-kvalifikationsdagen-14-augusti-2026-val-till-riksdagen.xlsx
- `old/scb-riksdag-results-2018.xlsx`: original user-provided SCB source, retained unchanged. Source page from the original script: https://www.scb.se/hitta-statistik/statistik-efter-amne/demokrati/allmanna-val/allmanna-val-valresultat/
- `old/jamforelser-2018-och-2022-valdistrikt-och-uppsamlingsdistrikt.xlsx`: original user-provided Valmyndigheten comparison file, retained unchanged.

The 2022 preliminary workbook is the completed preliminary count, not a sequence
of original election-night snapshots. A replay that reveals these counts does
not reproduce earlier revisions unless separately captured.

Archived 2022 preliminary and final machine-readable files and report timestamps
are now saved under `data/2022-feed/`. URLs, checksum verification, the two
corrected district records and timestamp limitations are recorded in
`data/processed/2022-report-times-SOURCE.md`. Replay uses the corrected preliminary
archive counts, not the two swapped Excel records. The final archive and final
workbook agree for every district/party.
