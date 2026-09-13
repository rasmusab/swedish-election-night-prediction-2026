"""Fast interpretable Gaussian-prior regression for election-night updates.

The ridge solution is a conditional Gaussian-prior MAP estimate, not an MCMC
posterior. Training uses ONLY revealed returns. Compositional predictions are
kept nonnegative and normalized. No actual missing-district volume is supplied.
"""
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import softmax
from val2026.election_data import PARTIES


def clr(shares):
    logs = np.log(np.maximum(shares, 1e-8))
    return logs - logs.mean(axis=1, keepdims=True)


def design_matrix(features, feature_penalty=20., county_penalty=10., municipality_penalty=10.):
    prior = features[['prior_' + p for p in PARTIES]].to_numpy(float)
    covariates = np.column_stack([prior, np.log(np.maximum(features.electorate, 1)),
                                 features.prior_rate, features.baseline_source.ne('district_mapping')])
    physical = features.kind.eq('ordinary').to_numpy()
    # For late pools, infer swings using ordinary-municipality covariates, then
    # apply those swings to the late pool's own historical share/volume offset.
    for municipality in features.municipality.unique():
        local = features.municipality.eq(municipality).to_numpy()
        if (local & ~physical).any():
            covariates[local & ~physical] = np.average(covariates[local & physical], axis=0,
                                                     weights=features.electorate.to_numpy()[local & physical])
    center = covariates[physical].mean(axis=0)
    scale = np.maximum(covariates[physical].std(axis=0), 1e-6)
    covariates = (covariates - center) / scale
    counties = pd.get_dummies(features.county, dtype=float)
    municipalities = pd.get_dummies(features.municipality, dtype=float)
    x = np.column_stack([np.ones(len(features)), covariates, counties, municipalities])
    penalty = np.r_[0.001, np.full(covariates.shape[1], feature_penalty),
                    np.full(counties.shape[1], county_penalty),
                    np.full(municipalities.shape[1], municipality_penalty)]
    names = ['intercept'] + ['prior_' + p for p in PARTIES] + ['log_electorate', 'prior_rate', 'unmatched']
    names += ['county_' + c for c in counties.columns] + ['municipality_' + c for c in municipalities.columns]
    return x, penalty, names


class SwingRegression:
    def __init__(self, features, *, transform='clr', feature_penalty=20.,
                 county_penalty=10., municipality_penalty=10., calibrate=False,
                 early_guard=False):
        self.features = features
        self.transform = transform
        self.calibrate = calibrate
        self.early_guard = early_guard
        self.x, self.penalty, self.names = design_matrix(features, feature_penalty, county_penalty, municipality_penalty)
        self.prior = features[['prior_' + p for p in PARTIES]].to_numpy(float)
        self.volume = features.expected_votes.to_numpy(float)

    def fit_predict(self, observed, sample_weight=None, return_details=False):
        started = perf_counter()
        known = np.isfinite(observed).all(axis=1)
        train = known & self.features.kind.eq('ordinary').to_numpy()
        if train.sum() < 2:
            raise ValueError('At least two reporting ordinary districts are required')
        counts = observed[train]
        totals = counts.sum(axis=1)
        if (totals <= 0).any():
            raise ValueError('A reporting district has no valid party votes')
        shares = (counts + 0.5) / (totals[:, None] + 0.5 * len(PARTIES))
        y_share = clr(shares) - clr(self.prior[train]) if self.transform == 'clr' else shares - self.prior[train]
        y_volume = np.log(totals / self.volume[train])
        y = np.column_stack([y_share, y_volume])
        weights = np.clip(np.sqrt(totals / 1000), 0.3, 2.0)
        if sample_weight is not None:
            weights *= np.asarray(sample_weight)[train]
        x = self.x[train]
        xtw = x.T * weights
        system = xtw @ x + np.diag(self.penalty)
        coef = cho_solve(cho_factor(system, check_finite=False), xtw @ y, check_finite=False)
        change = self.x @ coef
        if self.transform == 'clr':
            predicted_shares = softmax(clr(self.prior) + change[:, :len(PARTIES)], axis=1)
        else:
            predicted_shares = np.maximum(self.prior + change[:, :len(PARTIES)], 1e-6)
            predicted_shares /= predicted_shares.sum(axis=1, keepdims=True)
        volume = self.volume * np.exp(np.clip(change[:, -1], -1, 1))
        if self.calibrate:
            # Reconcile on the revealed sample only: correct retransformation
            # bias and ensure the fitted training aggregate tracks actual votes.
            fitted = (predicted_shares[train] * totals[:, None] * weights[:, None]).sum(axis=0)
            actual = (counts * weights[:, None]).sum(axis=0)
            adjustment = actual / np.maximum(fitted, 1e-9)
            predicted_shares *= adjustment
            predicted_shares /= predicted_shares.sum(axis=1, keepdims=True)
            volume *= np.sum(totals * weights) / np.sum(volume[train] * weights)
        ordinary = self.features.kind.eq('ordinary').to_numpy()
        volume[ordinary] = np.minimum(volume[ordinary], self.features.electorate.to_numpy()[ordinary])
        predicted = predicted_shares * volume[:, None]
        raw_prediction = predicted.copy()
        predicted[known] = observed[known]
        # Conservative startup: equal-weight county swing / regression through
        # 100 reports, then transition to full regression at 300. This limits extrapolation
        # from the very small, nonrandom set of first-reporting districts.
        blend = 0.5 + 0.5 * np.clip((train.sum() - 100) / 200, 0, 1) if self.early_guard else 1.0
        if blend < 1:
            from val2026.replay import simple_forecast
            conservative = simple_forecast(self.features, observed, 'county_swing')
            predicted = blend * predicted + (1 - blend) * conservative
        duration = perf_counter() - started
        if return_details:
            return predicted, {'seconds': duration, 'coef': coef, 'names': self.names,
                               'training_rows': int(train.sum()), 'raw_prediction': raw_prediction,
                               'regression_weight': float(blend)}
        return predicted
