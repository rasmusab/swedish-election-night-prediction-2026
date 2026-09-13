"""Synthetic arrival orders and scoring; final outcomes never enter predictors."""
import numpy as np
import pandas as pd
from val2026.election_data import DATA, PARTIES

FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
SCENARIOS = ['archived_report_time', 'random', 'small_first', 'geography_delayed', 'big_cities_late']


def load_backtest():
    d = pd.read_csv(DATA / 'processed/backtest-2018-2022.csv', dtype={'code': str, 'municipality': str, 'county': str})
    features = d.drop(columns=[c for c in d if c.startswith(('prelim_', 'final_'))])
    preliminary = d[['prelim_' + p for p in PARTIES]].to_numpy(float)
    final = d[['final_' + p for p in PARTIES]].to_numpy(float)
    return features, preliminary, final


def arrival_order(features, scenario, seed):
    """Stress scenarios use known electorate/geography, never future votes.

    Synthetic scenarios are NOT reconstructed actual 2022 reporting timestamps.
    archived_report_time uses the surviving archive's report/correction times,
    which may differ from initial reporting times on election night.
    Municipality collection pools are kept outstanding throughout election night.
    """
    rng = np.random.default_rng(seed)
    physical = np.flatnonzero(features.kind.eq('ordinary'))
    size = np.log(features.electorate.to_numpy()[physical])
    if scenario == 'archived_report_time':
        times = pd.read_csv(DATA / 'processed/2022-report-times.csv', dtype={'code': str}).set_index('code').reported_at
        reported_at = features.iloc[physical].code.map(times)
        if reported_at.isna().any():
            raise ValueError('Missing archived timestamps for ordinary districts')
        return physical[np.argsort(reported_at.to_numpy(), kind='stable')]
    if scenario == 'random':
        priority = rng.normal(size=len(physical))
    elif scenario == 'small_first':
        priority = size + rng.normal(0, 0.35, len(physical))
    elif scenario == 'geography_delayed':
        counties = sorted(features.county.unique())
        delays = dict(zip(counties, rng.normal(0, 1, len(counties))))
        priority = features.iloc[physical].county.map(delays).to_numpy() + 0.3 * size + rng.normal(0, 0.5, len(physical))
    elif scenario == 'big_cities_late':
        population = features[features.kind.eq('ordinary')].groupby('municipality').electorate.sum()
        municipal_size = np.log(features.iloc[physical].municipality.map(population).to_numpy())
        priority = 0.6 * municipal_size + 0.3 * size + rng.normal(0, 0.65, len(physical))
    else:
        raise ValueError(scenario)
    return physical[np.argsort(priority)]


def reveal(preliminary, order, fraction):
    """Missing districts remain NaN; model receives no unrevealed vote totals."""
    observed = np.full_like(preliminary, np.nan)
    ids = order[:max(1, int(np.ceil(len(order) * fraction)))]
    observed[ids] = preliminary[ids]
    return observed


def simple_forecast(features, observed, method):
    known = np.isfinite(observed).all(axis=1)
    historical = features[['prior_' + p for p in PARTIES]].to_numpy(float)
    volumes = features.expected_votes.to_numpy(float).copy()
    shares = historical.copy()
    if method == 'raw_reported':
        overall = observed[known].sum(axis=0)
        # Used only for national-share comparison; these are not volume estimates.
        return np.tile(overall / overall.sum(), (len(features), 1)) * volumes[:, None]
    if method in ['national_swing', 'county_swing']:
        current = observed[known].sum(axis=0)
        old = (historical[known] * volumes[known, None]).sum(axis=0)
        ratio = (current / current.sum()) / (old / old.sum())
        shares *= ratio
        volume_ratio = observed[known].sum() / volumes[known].sum()
        volumes *= volume_ratio
        if method == 'county_swing':
            for county in features.county.unique():
                local = features.county.eq(county).to_numpy()
                reported = local & known
                if reported.sum() < 10:
                    continue
                current_local = observed[reported].sum(axis=0)
                old_local = (historical[reported] * features.expected_votes.to_numpy()[reported, None]).sum(axis=0)
                local_ratio = (current_local / current_local.sum()) / (old_local / old_local.sum())
                blend = reported.sum() / (reported.sum() + 30)
                shares[local] = historical[local] * np.exp(blend * np.log(local_ratio) + (1 - blend) * np.log(ratio))
    shares /= shares.sum(axis=1, keepdims=True)
    predicted = shares * volumes[:, None]
    predicted[known] = observed[known]
    return predicted


def score(features, observed, predicted, final):
    truth = final.sum(axis=0)
    truth /= truth.sum()
    forecast = predicted.sum(axis=0)
    forecast /= forecast.sum()
    error = 100 * (forecast - truth)
    right = [PARTIES.index(p) for p in ['M', 'KD', 'L', 'SD']]
    left = [PARTIES.index(p) for p in ['S', 'V', 'MP', 'C']]
    known = np.isfinite(observed).all(axis=1)
    missing_physical = ~known & features.kind.eq('ordinary').to_numpy()
    metrics = {'mae_pp': np.abs(error).mean(), 'max_error_pp': np.abs(error).max(),
               'bloc_margin_error_pp': abs(error[right].sum() - error[left].sum()),
               'valid_vote_volume_error_pct': 100 * (predicted.sum() / final.sum() - 1)}
    for name, mask in [('remaining', missing_physical), ('comparable', missing_physical & features.baseline_source.eq('district_mapping').to_numpy()),
                       ('unmatched', missing_physical & ~features.baseline_source.eq('district_mapping').to_numpy())]:
        if mask.any():
            p = predicted[mask] / predicted[mask].sum(axis=1, keepdims=True)
            y = final[mask] / final[mask].sum(axis=1, keepdims=True)
            metrics[name + '_district_mae_pp'] = np.average(np.abs(p-y).mean(axis=1), weights=final[mask].sum(axis=1)) * 100
        else:
            metrics[name + '_district_mae_pp'] = np.nan
    return metrics, error
