"""Fast conditional predictive simulations around the frozen point regression.

The extra reporting, late-vote, and final-revision scales are explicit prior
assumptions, not variances learned from an unseen election result. Coefficient
draws already share county/municipality effects; no second set is added.
Reported preliminary counts and partial late counts are fixed in the
preliminary layer. Only a separate final-revision layer may change those counts.
"""
from dataclasses import asdict, dataclass, fields
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.special import softmax

from val2026.election_data import PARTIES
from val2026.forecast_model import clr


@dataclass(frozen=True)
class UncertaintyConfig:
    coefficient_uncertainty: bool = True
    district_residuals: bool = True
    reporting_discrepancy: bool = True
    late_effect: bool = True
    late_residuals: bool = True
    final_revisions: bool = True
    residual_prior_df: float = 40.0
    residual_prior_share_scale: float = 0.05
    residual_prior_log_volume_sd: float = 0.15
    reporting_discrepancy_multiplier: float = 0.5
    reporting_balance_floor: float = 0.25
    late_effect_multiplier: float = 1.0
    late_log_volume_sd_floor: float = 0.20
    partial_late_remaining_sd_fraction: float = 0.25
    revision_share_sd: float = 0.0005
    revision_log_volume_sd: float = 0.001
    revision_bound_sd: float = 3.0
    chunk_size: int = 16


@dataclass
class ConstituencyForecast:
    constituency_codes: list[str]
    vote_draws: np.ndarray
    preliminary_vote_draws: np.ndarray
    diagnostics: dict
    config: dict


def _configuration(config):
    if config is None:
        result = UncertaintyConfig()
    elif isinstance(config, UncertaintyConfig):
        result = config
    elif isinstance(config, dict):
        result = UncertaintyConfig(**config)
    else:
        raise ValueError('Config must be an UncertaintyConfig or a dictionary')
    for field in fields(result):
        value = getattr(result, field.name)
        if isinstance(field.default, bool):
            if not isinstance(value, bool):
                raise ValueError(f'{field.name} must be boolean')
        elif not np.isfinite(value) or value < 0:
            raise ValueError(f'{field.name} must be finite and nonnegative')
    if not isinstance(result.chunk_size, int) or result.chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    if result.residual_prior_df <= 0 or not 0 <= result.reporting_balance_floor <= 1:
        raise ValueError('Positive prior degrees of freedom and balance floor in [0, 1] required')
    return result


def _psd_factor(matrix):
    matrix = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(matrix)
    return vectors * np.sqrt(np.maximum(values, 0))[None, :]


def _reporting_balance(features, train, config):
    ordinary = features.kind.eq('ordinary').to_numpy()
    weights = features.electorate.to_numpy(float)

    def tv(categories):
        return float(sum(abs(weights[ordinary & (categories == group)].sum() / weights[ordinary].sum()
                             - weights[train & (categories == group)].sum() / weights[train].sum())
                         for group in np.unique(categories[ordinary])) / 2)

    county_tv = tv(features.county.to_numpy())
    # Municipality population distinguishes delayed cities from smaller places
    # inside the same county, even if their polling districts have similar size.
    municipal_sizes = features.loc[ordinary].groupby('municipality').electorate.sum()
    population = features.municipality.map(municipal_sizes).to_numpy(float)
    cuts = np.quantile(population[ordinary], [0.2, 0.4, 0.6, 0.8])
    population_tv = tv(np.searchsorted(cuts, population))
    log_size = np.log(np.maximum(features.expected_votes.to_numpy(float), 1))
    center = np.average(log_size[ordinary], weights=weights[ordinary])
    spread = np.sqrt(np.average((log_size[ordinary] - center) ** 2, weights=weights[ordinary]))
    size_gap = abs(np.average(log_size[train], weights=weights[train]) - center) / max(spread, 1e-8)
    taper = max(config.reporting_balance_floor, min(1., 2 * max(county_tv, population_tv) + 0.5 * size_gap))
    return {'county_total_variation': county_tv, 'municipality_population_total_variation': population_tv,
            'district_log_size_standardized_gap': float(size_gap), 'taper': float(taper)}


def simulate_constituency_votes(model, observed, *, predicted=None, partial_late=None,
                                draws=1000, seed=2026, config=None):
    """Return joint final/preliminary vote draws, aggregating districts in chunks.

    No target-election final or unrevealed vote counts are accepted. The response
    covariance has a historical-share prior plus revealed ordinary residuals.
    Gaussian coefficient covariance is H^-1 x Sigma conditional on fixed ridge
    penalties; residual degrees of freedom use the ridge smoother. Nonlinear
    positivity/turnout constraints can move simulation means away from the point
    forecast, which remains separate and is never used to recenter aggregates.

    This is an approximate posterior predictive sensitivity model, not a claim
    that its probability levels have been calibrated on independent elections.
    """
    started = perf_counter()
    cfg = _configuration(config)
    if not isinstance(draws, int) or draws < 2:
        raise ValueError('At least two integer draws are required')
    features = model.features
    n, parties = len(features), len(PARTIES)
    observed = np.asarray(observed, float)
    if observed.shape != (n, parties):
        raise ValueError('Observed counts must align with features and parties')
    known = np.isfinite(observed).all(axis=1)
    if np.any(~known & ~np.isnan(observed).all(axis=1)) or (observed[known] < 0).any():
        raise ValueError('Each district must have nonnegative finite counts or all missing counts')
    if 'constituency' not in features or features.constituency.isna().any():
        raise ValueError('Every feature row needs a constituency')
    codes = features.constituency.astype(str).to_numpy()
    if any(not code or code == 'nan' for code in codes):
        raise ValueError('Constituency codes cannot be empty')
    constituency_codes = sorted(set(codes))
    group = np.array([constituency_codes.index(code) for code in codes])
    ordinary = features.kind.eq('ordinary').to_numpy()
    if not features.kind.isin(['ordinary', 'late']).all():
        raise ValueError('District kinds must be ordinary or late')
    train = known & ordinary
    electorate = features.electorate.to_numpy(float)
    if not np.isfinite(electorate).all() or (electorate[ordinary] <= 0).any():
        raise ValueError('Ordinary electorate must be positive and finite')
    if train.sum() < 2:
        raise ValueError('At least two reporting ordinary districts are required')
    base, fitted = model.fit_predict(observed, return_details=True)
    predicted = base if predicted is None else np.asarray(predicted, float).copy()
    if predicted.shape != observed.shape or not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError('Predictions must be aligned, finite, and nonnegative')
    if not np.allclose(predicted[known], observed[known], rtol=0, atol=1e-7):
        raise ValueError('Predictions must preserve completed preliminary counts')
    # The frozen pre-election electorate is an estimate bound for outstanding
    # districts, not a reason to reject or cap verified votes already reported.
    if (predicted[ordinary & ~known].sum(axis=1) > electorate[ordinary & ~known] + 1e-6).any():
        raise ValueError('Ordinary predictions cannot exceed the electorate')
    partial = np.zeros_like(observed) if partial_late is None else np.asarray(partial_late, float)
    if partial.shape != observed.shape or not np.isfinite(partial).all() or (partial < 0).any():
        raise ValueError('Partial late counts must be aligned, finite, and nonnegative')
    if (partial[ordinary] != 0).any() or (partial[known] > 0).any():
        raise ValueError('Partial counts belong only to incomplete late pools')
    # The simulation can also be called directly before the runtime applies its
    # partial-count bound. Adjust only those incomplete late point rows here.
    predicted = np.maximum(predicted, partial)

    counts = observed[train]
    totals = counts.sum(axis=1)
    shares = (counts + 0.5) / (totals[:, None] + 0.5 * parties)
    y_share = clr(shares) - clr(model.prior[train]) if model.transform == 'clr' else shares - model.prior[train]
    response = np.column_stack([y_share, np.log(totals / model.volume[train])])
    x = model.x[train]
    weights = np.clip(np.sqrt(totals / 1000), 0.3, 2.)
    gram = (x.T * weights) @ x
    system = gram + np.diag(model.penalty)
    coefficient_covariance = cho_solve(cho_factor(system, check_finite=False),
                                     np.eye(system.shape[0]), check_finite=False)
    coefficient_covariance = (coefficient_covariance + coefficient_covariance.T) / 2
    smoother = coefficient_covariance @ gram
    effective_parameters = float(np.trace(smoother))
    residual_df = max(0., float(len(x) - 2 * effective_parameters + np.trace(smoother @ smoother)))
    residual = response - x @ fitted['coef']
    scatter = (residual.T * weights) @ residual
    historical_shares = np.average(model.prior, axis=0, weights=model.volume)
    historical_shares /= historical_shares.sum()
    prior_share_covariance = cfg.residual_prior_share_scale ** 2 * (
        np.diag(historical_shares) - np.outer(historical_shares, historical_shares))
    if model.transform == 'clr':
        derivative = (np.eye(parties) - np.ones((parties, parties)) / parties) @ np.diag(1 / np.maximum(historical_shares, 1e-8))
        prior_share_covariance = derivative @ prior_share_covariance @ derivative.T
    prior_covariance = np.zeros((parties + 1, parties + 1))
    prior_covariance[:parties, :parties] = prior_share_covariance
    prior_covariance[-1, -1] = cfg.residual_prior_log_volume_sd ** 2
    response_covariance = (scatter + cfg.residual_prior_df * prior_covariance) / (residual_df + cfg.residual_prior_df)
    response_factor = _psd_factor(response_covariance)
    coefficient_factor = cholesky(coefficient_covariance, lower=True, check_finite=False)
    balance = _reporting_balance(features, train, cfg)

    missing = ~known
    missing_ordinary = ordinary[missing]
    free_map = np.maximum(predicted[missing] - partial[missing], 0)
    free_volume = free_map.sum(axis=1)
    free_shares = np.divide(free_map, free_volume[:, None], out=model.prior[missing].copy(), where=free_volume[:, None] > 0)
    # An incomplete pool whose counted floor already consumes its point
    # prediction may still receive votes. Use an explicit one-sided allowance
    # for this unresolved remainder rather than declaring its count complete.
    partial_remaining_sd = np.where(
        (~missing_ordinary) & (partial[missing].sum(axis=1) > 0),
        np.maximum(0., cfg.partial_late_remaining_sd_fraction * model.volume[missing] - free_volume), 0.)
    prediction_weights = np.clip(np.sqrt(free_volume / 1000), 0.3, 2.)
    fixed = np.zeros((len(constituency_codes), parties))
    np.add.at(fixed, group[known], observed[known])
    np.add.at(fixed, group[missing], partial[missing])
    preliminary = np.broadcast_to(fixed, (draws, *fixed.shape)).copy()
    rng = np.random.default_rng(seed)
    # Each common vector is shared across its entire group of outstanding rows.
    selection = rng.normal(size=(draws, parties + 1)) @ response_factor.T
    selection *= cfg.reporting_discrepancy_multiplier * balance['taper'] if cfg.reporting_discrepancy else 0
    late = (rng.normal(size=(draws, parties + 1)) @ response_factor.T) * cfg.late_effect_multiplier
    late_extra_volume_sd = np.sqrt(max(0., cfg.late_log_volume_sd_floor ** 2
                                      - response_covariance[-1, -1] * cfg.late_effect_multiplier ** 2))
    late[:, -1] += rng.normal(size=draws) * late_extra_volume_sd
    if not cfg.late_effect:
        late[:] = 0
    coefficient_count = model.x.shape[1]
    response_count = parties + 1
    clipped_share_values = 0
    capped_ordinary_volumes = 0
    for start in range(0, draws, cfg.chunk_size):
        count = min(cfg.chunk_size, draws - start)
        noise = np.zeros((missing.sum(), count, response_count))
        if cfg.coefficient_uncertainty:
            coef = coefficient_factor @ rng.normal(size=(coefficient_count, count * response_count))
            coef = (coef.reshape(coefficient_count * count, response_count) @ response_factor.T).reshape(coefficient_count, count * response_count)
            noise += (model.x[missing] @ coef).reshape(missing.sum(), count, response_count)
        noise[missing_ordinary] += selection[None, start:start + count]
        noise[~missing_ordinary] += late[None, start:start + count]
        residual_rows = ((missing_ordinary & cfg.district_residuals)
                         | (~missing_ordinary & cfg.late_residuals))
        if residual_rows.any():
            independent = rng.normal(size=(residual_rows.sum(), count, response_count)) @ response_factor.T
            noise[residual_rows] += independent / np.sqrt(prediction_weights[residual_rows])[:, None, None]
        if model.transform == 'clr':
            simulated_shares = softmax(np.log(np.maximum(free_shares, 1e-12))[:, None, :] + noise[:, :, :parties], axis=2)
        else:
            simulated_shares = free_shares[:, None, :] + noise[:, :, :parties]
            clipped_share_values += int((simulated_shares < 0).sum())
            simulated_shares = np.maximum(simulated_shares, 0)
            simulated_shares /= simulated_shares.sum(axis=2, keepdims=True)
        simulated_volumes = free_volume[:, None] * np.exp(np.clip(noise[:, :, -1], -3, 3))
        if cfg.late_residuals and np.any(partial_remaining_sd):
            simulated_volumes += np.abs(rng.normal(size=simulated_volumes.shape)) * partial_remaining_sd[:, None]
        capped_ordinary_volumes += int((simulated_volumes[missing_ordinary] > electorate[missing][missing_ordinary, None]).sum())
        simulated_volumes[missing_ordinary] = np.minimum(simulated_volumes[missing_ordinary], electorate[missing][missing_ordinary, None])
        district_counts = simulated_shares * simulated_volumes[:, :, None]
        grouped = np.zeros((len(constituency_codes), count, parties))
        np.add.at(grouped, group[missing], district_counts)
        preliminary[start:start + count] += grouped.transpose(1, 0, 2)

    final = preliminary.copy()
    revision_bounded_values = 0
    revision_clipped_shares = 0
    if cfg.final_revisions:
        # Symmetric national revision effects preserve party dependence. Bounds
        # are explicit assumptions, not observed limits of possible corrections.
        z = rng.normal(size=(draws, parties + 1))
        z[:, :parties] -= z[:, :parties].mean(axis=1, keepdims=True)
        revision_bounded_values = int((np.abs(z) > cfg.revision_bound_sd).sum())
        z = np.clip(z, -cfg.revision_bound_sd, cfg.revision_bound_sd)
        z[:, :parties] -= z[:, :parties].mean(axis=1, keepdims=True)
        final_volume = preliminary.sum(axis=2)
        final_shares = np.divide(preliminary, final_volume[:, :, None], out=np.zeros_like(preliminary), where=final_volume[:, :, None] > 0)
        final_shares += z[:, None, :parties] * cfg.revision_share_sd
        revision_clipped_shares = int((final_shares < 0).sum())
        final_shares = np.maximum(final_shares, 0)
        divisor = final_shares.sum(axis=2, keepdims=True)
        final_shares = np.divide(final_shares, divisor, out=np.zeros_like(final_shares), where=divisor > 0)
        final = final_shares * (final_volume * np.exp(z[:, None, -1] * cfg.revision_log_volume_sd))[:, :, None]

    national_point = predicted.sum(axis=0)
    national_final = final.sum(axis=1)
    national_shares = national_final / national_final.sum(axis=1, keepdims=True)
    national_preliminary = preliminary.sum(axis=1)
    preliminary_shares = national_preliminary / national_preliminary.sum(axis=1, keepdims=True)
    unseen = set(features.loc[missing, 'municipality']) - set(features.loc[train, 'municipality'])
    diagnostics = {
        'draws': draws, 'seed': int(seed), 'training_rows': int(train.sum()),
        'missing_ordinary_rows': int((missing & ordinary).sum()),
        'missing_late_rows': int((missing & ~ordinary).sum()),
        'unseen_municipalities': len(unseen), 'effective_parameters': effective_parameters,
        'residual_df': residual_df,
        'residual_prior_weight': cfg.residual_prior_df / (residual_df + cfg.residual_prior_df),
        'response_covariance': response_covariance.tolist(),
        'historical_share_prior': historical_shares.tolist(),
        'reporting_balance': balance,
        'reported_preliminary_counts_held_fixed': True,
        'partial_late_counted_votes_held_fixed': float(partial.sum()),
        'partial_late_pools_with_one_sided_remainder': int((partial_remaining_sd > 0).sum()) if cfg.late_residuals else 0,
        'aggregate_recentering': False,
        'negative_district_share_values_clipped': clipped_share_values,
        'ordinary_turnout_draws_capped': capped_ordinary_volumes,
        'revision_standard_normal_values_bounded': revision_bounded_values,
        'negative_revision_share_values_clipped': revision_clipped_shares,
        'max_draw_mean_offset_pp': float(np.max(np.abs(national_shares.mean(axis=0) - national_point / national_point.sum())) * 100),
        'max_preliminary_draw_mean_offset_pp': float(np.max(np.abs(preliminary_shares.mean(axis=0) - national_point / national_point.sum())) * 100),
        'regression_weight': fitted['regression_weight'],
        'seconds': perf_counter() - started,
        'scope': ('Approximate conditional Bayesian predictive simulations with fixed ridge penalties. '
                  'Residual covariance combines revealed ordinary returns and a historical-share prior. '
                  'Reporting-selection, shared late-vote, and final-revision scales are explicit sensitivity '
                  'assumptions; probability levels are not calibrated across independent elections. '
                  'Early guarded point forecasts retain the full regression covariance. '
                  'The final-revision layer can change completed preliminary counts; the preliminary layer cannot.'),
    }
    if not np.isfinite(final).all() or (final < 0).any():
        raise ArithmeticError('Predictive vote draws must be finite and nonnegative')
    return ConstituencyForecast(constituency_codes, final, preliminary, diagnostics, asdict(cfg))
