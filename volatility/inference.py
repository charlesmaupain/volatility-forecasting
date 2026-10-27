"""Paired dependent-data uncertainty, conditional on saved forecast paths."""
import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap
from .experiment import loss


def paired_losses(predictions, reference='HAR'):
    required = ['origin', 'target_end', 'model', 'observed', 'forecast']
    if predictions[required].isna().any().any():
        raise ValueError('Missing forecast values')
    if predictions.duplicated(['origin', 'model']).any():
        raise ValueError('Duplicate model/origin forecast')
    for name in ['target_end', 'observed']:
        if predictions.groupby('origin')[name].nunique().max() != 1:
            raise ValueError('Models must share outcomes and target dates')
    if not np.isfinite(predictions[['observed', 'forecast']].to_numpy()).all() or (predictions[['observed', 'forecast']] <= 0).any().any():
        raise ValueError('Observed and forecast variances must be positive and finite')
    values = predictions.assign(qlike=loss(predictions.observed, predictions.forecast))
    wide = values.pivot(index='origin', columns='model', values='qlike').sort_index()
    if wide.isna().any().any() or reference not in wide:
        raise ValueError('Models must share every forecast origin')
    return wide.drop(columns=reference).subtract(wide[reference], axis=0), values


def bootstrap_intervals(differences, block_length=20, repetitions=5000, seed=271828, confidence=.95):
    matrix = np.asarray(differences, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError('Require a finite matrix of paired differences')
    if not (1 <= block_length <= len(matrix)) or repetitions < 100 or not (0 < confidence < 1):
        raise ValueError('Invalid bootstrap settings')
    bootstrap = StationaryBootstrap(block_length, matrix, seed=seed)
    means = bootstrap.apply(lambda sample: sample.mean(axis=0), repetitions)
    estimate = matrix.mean(axis=0)
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail], axis=0)
    # Same sampled dates for every column: approximate joint coverage for this family.
    max_deviation = np.max(np.abs(means - estimate), axis=1)
    radius = np.quantile(max_deviation, confidence)
    return pd.DataFrame({'model': differences.columns, 'mean_difference': estimate,
                         'pointwise_low': low, 'pointwise_high': high,
                         'family_low': estimate - radius, 'family_high': estimate + radius,
                         'block_length': block_length, 'repetitions': repetitions})


def causal_regimes(series, origins, quantiles=(.25, .75)):
    weekly = series.rolling(5).mean()
    prior = weekly.shift(1)
    lower = prior.expanding(min_periods=252).quantile(quantiles[0])
    upper = prior.expanding(min_periods=252).quantile(quantiles[1])
    frame = pd.DataFrame({'weekly': weekly, 'lower': lower, 'upper': upper}).loc[origins]
    if frame.isna().any().any():
        raise ValueError('Insufficient past data for regimes')
    frame['regime'] = np.where(frame.weekly < frame.lower, 'Low',
                               np.where(frame.weekly > frame.upper, 'High', 'Middle'))
    return frame



def floor_mask(forecasts, floor):
    """Count a float32-clipped floor after conversion to float64/CSV."""
    return np.isclose(np.asarray(forecasts), floor, rtol=1e-6, atol=0)
