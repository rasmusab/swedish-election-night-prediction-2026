"""Conditional Gaussian uncertainty for the fitted swing regression.

This is an empirical, conditional Gaussian-prior approximation, not an MCMC
posterior or a calibrated election-winning probability. It conditions on fixed
ridge penalties, observed preliminary counts, and historical late-vote offsets.
It excludes recount revisions and an additional election-wide late-vote shift.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.special import softmax

from val2026.election_data import PARTIES
from val2026.forecast_model import clr


SCOPE = (
    'Conditional 90% intervals: fixed shrinkage, empirical Gaussian errors, '
    'reported preliminary counts held fixed. Includes joint share-volume '
    'coefficient uncertainty and ordinary-district residual variation. '
    'Excludes recount revisions, extra election-wide late-vote shifts, '
    'hyperparameter uncertainty, and model misspecification.'
)


def interval_scope(late_shift):
    if not late_shift:
        return SCOPE
    return ('Conditional 90% intervals with an assumed shared national late-vote '
            'effect, using one reported-district residual covariance as its prior '
            'scale. Includes joint share-volume coefficient uncertainty and '
            'ordinary-district residual variation. Reported preliminary counts '
            'remain fixed. Excludes recount revisions, hyperparameter uncertainty '
            'and model misspecification. Late-effect variance is an assumption, '
            'not identified by ordinary returns.')


@dataclass
class ConditionalForecast:
    map_shares: np.ndarray
    share_draws: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    diagnostics: dict


def conditional_share_draws(model, observed, *, draws=200, seed=2026,
                            chunk_size=16, recenter=True, late_shift=False):
    """Simulate national shares using only features and revealed observations.

    The coefficient covariance is H^-1 (across features) times Sigma (across
    party changes and log volume), with H = X'WX + penalty. Sigma is estimated
    from revealed residuals using n - 2 tr(A) + tr(A^2) residual degrees of
    freedom for the weighted ridge smoother. County and municipality columns
    share coefficient draws across all districts, including unseen areas.

    Residual vectors preserve party-volume covariance and are added only to
    unreported ordinary districts. Late pools inherit regression uncertainty,
    but no independently invented late-specific residual distribution. Draws
    are recentered on MAP *missing vote counts* before adding fixed reported
    counts, preventing retransformation from silently changing the point model.
    Neither final results nor unrevealed preliminary results are accepted.
    """
    started = perf_counter()
    if draws < 2 or chunk_size < 1:
        raise ValueError('At least two draws and a positive chunk size required')
    observed = np.asarray(observed, dtype=float)
    if observed.shape != (len(model.features), len(PARTIES)):
        raise ValueError('Observed counts must align with features and parties')
    known = np.isfinite(observed).all(axis=1)
    if np.any(np.isfinite(observed).any(axis=1) & ~known):
        raise ValueError('A district must have all counts or all missing counts')
    if (observed[known] < 0).any():
        raise ValueError('Reported counts cannot be negative')
    ordinary = model.features.kind.eq('ordinary').to_numpy()
    train = known & ordinary
    predicted, fitted = model.fit_predict(observed, return_details=True)
    map_total = predicted.sum(axis=0)
    map_shares = map_total / map_total.sum()

    counts = observed[train]
    totals = counts.sum(axis=1)
    shares = (counts + 0.5) / (totals[:, None] + 0.5 * len(PARTIES))
    y_shares = (clr(shares) - clr(model.prior[train]) if model.transform == 'clr'
                else shares - model.prior[train])
    y = np.column_stack([y_shares, np.log(totals / model.volume[train])])
    x = model.x[train]
    weights = np.clip(np.sqrt(totals / 1000), 0.3, 2.0)
    gram = (x.T * weights) @ x
    system = gram + np.diag(model.penalty)
    factor = cho_factor(system, check_finite=False)
    covariance = cho_solve(factor, np.eye(system.shape[0]), check_finite=False)
    covariance = (covariance + covariance.T) / 2
    smoother = covariance @ gram
    effective_parameters = float(np.trace(smoother))
    residual_df = float(len(x) - 2 * effective_parameters + np.trace(smoother @ smoother))
    residual = y - x @ fitted['coef']
    response_covariance = (residual.T * weights) @ residual / max(residual_df, 1.)
    response_covariance = (response_covariance + response_covariance.T) / 2
    # Party-share changes sum to zero, so the joint covariance is singular by
    # construction. An eigendecomposition preserves that compositional constraint.
    eigvals, eigvecs = np.linalg.eigh(response_covariance)
    response_factor = eigvecs * np.sqrt(np.maximum(eigvals, 0))[None, :]
    coefficient_factor = cholesky(covariance, lower=True, check_finite=False)

    missing = ~known
    missing_ordinary = ordinary[missing]
    missing_map = predicted[missing]
    missing_volumes = missing_map.sum(axis=1)
    missing_shares = missing_map / np.maximum(missing_volumes[:, None], 1e-12)
    prediction_weights = np.clip(np.sqrt(missing_volumes / 1000), 0.3, 2.0)
    missing_x = model.x[missing]
    electorate = model.features.electorate.to_numpy(float)[missing]
    fixed_counts = observed[known].sum(axis=0)
    simulated_missing = np.zeros((draws, len(PARTIES)))
    rng = np.random.default_rng(seed)
    ncoef, nresponse = fitted['coef'].shape
    for start in range(0, draws, chunk_size):
        count = min(chunk_size, draws - start)
        # Each draw gets one coefficient matrix shared by every future district.
        coef_noise = coefficient_factor @ rng.normal(size=(ncoef, count * nresponse))
        coef_noise = (coef_noise.reshape(ncoef * count, nresponse)
                      @ response_factor.T).reshape(ncoef, count * nresponse)
        noise = (missing_x @ coef_noise).reshape(len(missing_map), count, nresponse)
        if late_shift and (~missing_ordinary).any():
            # A shared national late-vote change is unidentifiable from ordinary
            # returns. This explicit prior uses one district residual covariance
            # as its scale, jointly across parties and volume. It is a sensitivity
            # assumption, not an estimated or validated late-effect variance.
            common_late = rng.normal(size=(count, nresponse)) @ response_factor.T
            noise[~missing_ordinary] += common_late[None, :, :]
        if missing_ordinary.any():
            residual_noise = rng.normal(size=(missing_ordinary.sum(), count, nresponse))
            residual_noise = residual_noise @ response_factor.T
            noise[missing_ordinary] += (residual_noise /
                np.sqrt(prediction_weights[missing_ordinary])[:, None, None])
        if model.transform == 'clr':
            simulated_shares = softmax(np.log(np.maximum(missing_shares, 1e-12))[:, None, :]
                                       + noise[:, :, :len(PARTIES)], axis=2)
        else:
            simulated_shares = np.maximum(missing_shares[:, None, :]
                                           + noise[:, :, :len(PARTIES)], 1e-8)
            simulated_shares /= simulated_shares.sum(axis=2, keepdims=True)
        simulated_volumes = missing_volumes[:, None] * np.exp(np.clip(noise[:, :, -1], -3, 3))
        simulated_volumes[missing_ordinary] = np.minimum(
            simulated_volumes[missing_ordinary], electorate[missing_ordinary, None])
        simulated_missing[start:start + count] = (
            simulated_shares * simulated_volumes[:, :, None]).sum(axis=0)

    if recenter:
        simulated_missing += missing_map.sum(axis=0) - simulated_missing.mean(axis=0)
    clipped_values = int((simulated_missing < 0).sum())
    simulated_missing = np.maximum(simulated_missing, 0)
    national_counts = simulated_missing + fixed_counts
    share_draws = national_counts / national_counts.sum(axis=1, keepdims=True)
    lower, upper = np.quantile(share_draws, [0.05, 0.95], axis=0)
    unseen = set(model.features.loc[missing, 'municipality']) - set(model.features.loc[train, 'municipality'])
    diagnostics = {
        'draws': draws, 'training_rows': int(train.sum()),
        'missing_ordinary_rows': int((missing & ordinary).sum()),
        'missing_late_rows': int((missing & ~ordinary).sum()),
        'unseen_municipalities': len(unseen),
        'effective_parameters': effective_parameters, 'residual_df': residual_df,
        'recenter_missing_counts_on_map': recenter,
        'negative_aggregate_values_clipped': clipped_values,
        'max_draw_mean_offset_pp': float(np.max(np.abs(share_draws.mean(axis=0) - map_shares)) * 100),
        'seconds': perf_counter() - started, 'scope': interval_scope(late_shift),
        'shared_late_shift_prior': bool(late_shift),
        'late_prior_scale': 'one reported-district residual covariance' if late_shift else None,
    }
    assert np.isfinite(share_draws).all() and np.allclose(share_draws.sum(axis=1), 1)
    return ConditionalForecast(map_shares, share_draws, lower, upper, diagnostics)
