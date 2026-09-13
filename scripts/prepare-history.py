# %%
"""Steps 1–2: historical baselines for 2026 and the 2018→2022 backtest.

Run: uv run scripts/prepare-history.py
Inputs are the downloaded official workbooks and the original 2018 files.
Only 2022 outcomes, not 2026 rehearsal outcomes, enter the 2026 baseline.
"""
import json
import pandas as pd
from val2026.election_data import DATA, PARTIES, results_2018, results_2022, mapping_2022, mapping_2026, build_baseline, target_2022

# %% Final and preliminary 2022 results, including all valid minor-party votes.
r2022 = results_2022()
print('2022 final valid votes:', int(r2022[['final_' + p for p in PARTIES]].sum().sum()))
print('2022 preliminary valid votes:', int(r2022[['prelim_' + p for p in PARTIES]].sum().sum()))

# %% Real 2026 pre-election electorate; late pools use municipality electorate.
electorate = pd.read_excel(DATA / 'electorate-2026.xlsx', sheet_name='rostber_per_distrikt', dtype={'Valdistriktskod': str, 'Kommunkod': str})
electorate = electorate.rename(columns={'Valdistriktskod': 'code', 'Kommunkod': 'municipality', 'Valdistrikt': 'name', 'Totalt': 'electorate'})
electorate = electorate.dropna(subset=['code']).copy()  # Exclude the national total/footer.
electorate['county'] = electorate.code.str[:2]
electorate['kind'] = 'ordinary'
target = electorate[['code', 'municipality', 'county', 'name', 'electorate', 'kind']]
late = target.groupby(['municipality', 'county'], as_index=False).electorate.sum()
late['code'] = late.municipality + 'LATE'
late['kind'] = 'late'
late['name'] = 'Municipality late-vote pool'
target = pd.concat([target, late], ignore_index=True)
prior2022 = r2022.rename(columns={'final_' + p: p for p in PARTIES})
baseline2026 = build_baseline(prior2022, target, mapping_2026())
baseline2026.to_csv(DATA / 'processed/baseline-2026.csv', index=False)
print('\n2026 baseline sources:')
print(baseline2026.groupby(['kind', 'baseline_source']).size().to_string())

# %% Backtest features use only 2018 votes and 2022 pre-election geography/size.
# The electorate column is taken from the results workbook, but eligibility was
# determined before election day. It is a covariate, not a revealed vote total.
observations = target_2022(r2022)
features = observations[['code', 'municipality', 'county', 'name', 'electorate', 'kind']]
baseline2022 = build_baseline(results_2018(), features, mapping_2022())
observed_columns = ['code'] + [phase + '_' + p for phase in ['prelim', 'final'] for p in PARTIES]
backtest = baseline2022.merge(observations[observed_columns], validate='one_to_one')
backtest.to_csv(DATA / 'processed/backtest-2018-2022.csv', index=False)
print('\nBacktest baseline sources:')
print(backtest.groupby(['kind', 'baseline_source']).size().to_string())
summary = {
    '2026_baseline_units': len(baseline2026),
    '2026_ordinary': int(baseline2026.kind.eq('ordinary').sum()),
    '2026_late_municipality_pools': int(baseline2026.kind.eq('late').sum()),
    '2022_backtest_units': len(backtest),
    '2022_final_valid_votes': int(r2022[['final_' + p for p in PARTIES]].sum().sum()),
    '2022_preliminary_valid_votes': int(r2022[['prelim_' + p for p in PARTIES]].sum().sum()),
    '2026_fallback_counts': baseline2026.baseline_source.value_counts().to_dict(),
    '2022_fallback_counts': backtest.baseline_source.value_counts().to_dict(),
}
(DATA / 'processed/preparation-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('\nSaved baselines and backtest table under data/processed/.')
