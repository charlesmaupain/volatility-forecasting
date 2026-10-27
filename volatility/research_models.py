"""Research extensions with explicitly matured residuals and return state."""
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.linear_model import LinearRegression
from .experiment import FEATURES, training_at, predict, xgb_model


def har_oof(frame, cfg, start, min_history=500):
    """Historical monthly HAR predictions, each fitted without its outcome."""
    rows = []
    for _, block in frame.loc[start:].groupby(frame.loc[start:].index.to_period('M')):
        origin = block.index[0]
        training = training_at(frame, origin)
        if len(training) < min_history:
            continue
        model = LinearRegression().fit(training[FEATURES].to_numpy() * cfg['scale'],
                                       training.target.to_numpy() * cfg['scale'])
        values = predict(model, block, cfg)
        for (date, row), forecast in zip(block.iterrows(), values):
            rows.append({'origin': date, 'target_end': row.target_end,
                         'base_fit_origin': origin, 'base_training_target_max': training.target_end.max(),
                         'har_forecast': float(forecast), 'target': float(row.target),
                         **{name: float(row[name]) for name in FEATURES}})
    if not rows:
        raise ValueError('No historical out-of-sample HAR predictions')
    return pd.DataFrame(rows).set_index('origin')


def matured_residuals(oof, origin):
    eligible = oof.loc[(oof.index < origin) & (oof.target_end <= origin)].copy()
    if not ((eligible.base_training_target_max <= eligible.base_fit_origin)
            & (eligible.base_fit_origin <= eligible.index)
            & (eligible.index < eligible.target_end)).all():
        raise ValueError('Invalid out-of-sample residual lineage')
    eligible['residual'] = eligible.target - eligible.har_forecast
    return eligible


def trailing_returns(returns, origin, minimum):
    """Never join observations across an unavailable trading session."""
    known = returns.loc[:origin]
    if len(known) == 0 or known.index[-1] != origin or pd.isna(known.iloc[-1]):
        raise ValueError('Current session return unavailable')
    missing = np.flatnonzero(known.isna().to_numpy())
    if len(missing):
        known = known.iloc[missing[-1] + 1:]
    if len(known) < minimum or not np.isfinite(known).all():
        raise ValueError('Insufficient contiguous return history')
    return known


def variance_step(omega, alpha, beta, current_return, current_variance):
    value = omega + alpha * current_return ** 2 + beta * current_variance
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Invalid GARCH state')
    return float(value)


def garch_block(returns, origins, minimum=252):
    origin = origins[0]
    training = trailing_returns(returns, origin, minimum) * 100.0
    model = arch_model(training, mean='Zero', vol='GARCH', p=1, q=1,
                       dist='StudentsT', rescale=False)
    fitted = model.fit(disp='off', show_warning=False, options={'maxiter': 2000})
    if fitted.convergence_flag != 0:
        raise RuntimeError(f'GARCH failed at {origin}: {fitted.optimization_result.message}')
    params = fitted.params
    omega, alpha, beta = (float(params[k]) for k in ['omega', 'alpha[1]', 'beta[1]'])
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta > 1.000001:
        raise ValueError('Invalid GARCH parameter constraints')
    state = float(fitted.conditional_volatility.iloc[-1] ** 2)
    first_expected = float(fitted.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]) / 10000
    outputs = []
    previous_location = None
    for date in origins:
        location = returns.index.get_loc(date)
        if previous_location is not None and location != previous_location + 1:
            raise ValueError('GARCH prediction block must contain consecutive sessions')
        current = float(returns.loc[date]) * 100
        state = variance_step(omega, alpha, beta, current, state)
        outputs.append(state / 10000)
        previous_location = location
    np.testing.assert_allclose(outputs[0], first_expected, rtol=1e-10)
    return np.asarray(outputs), {
        'return_training_start': str(training.index.min().date()),
        'return_training_end': str(training.index.max().date()),
        'return_training_count': len(training), 'convergence_flag': int(fitted.convergence_flag),
        'omega_percent_squared': omega, 'alpha': alpha, 'beta': beta,
        'persistence': alpha + beta, 'nu': float(params['nu'])
    }


def extension_predictions(frame, returns, baseline, original_fits, cfg, research, progress=True):
    evaluation_origins = pd.DatetimeIndex(sorted(baseline.origin.unique()))
    evaluation = frame.loc[evaluation_origins]
    if evaluation.target_end.max() >= pd.Timestamp(cfg['holdout_start']):
        raise ValueError('Final holdout boundary crossed')
    oof = har_oof(frame, cfg, research['oof_start'])
    reference_har = baseline[baseline.model == 'HAR'].set_index('origin').forecast
    np.testing.assert_allclose(oof.loc[reference_har.index].har_forecast, reference_har, rtol=1e-10)
    fit_lookup = {r['fit_origin']: r for r in original_fits}
    rows, fit_logs = [], []
    for _, block in evaluation.groupby(evaluation.index.to_period('M')):
        origin = block.index[0]
        original_fit = fit_lookup[str(origin.date())]
        eligible = training_at(frame, origin)
        rolling = eligible.tail(research['rolling_observations'])
        if len(rolling) != research['rolling_observations']:
            raise ValueError('Insufficient rolling history')
        x, y = rolling[FEATURES].to_numpy() * cfg['scale'], rolling.target.to_numpy() * cfg['scale']
        har = LinearRegression().fit(x, y)
        xgb = xgb_model(original_fit['xgb_parameters'], cfg).fit(x, y)
        residuals = matured_residuals(oof, origin)
        if len(residuals) < research['hybrid']['min_residuals']:
            raise ValueError('Insufficient matured out-of-sample residuals')
        hybrid_cfg = dict(cfg, learning_rate=research['hybrid']['learning_rate'])
        hybrid = xgb_model({k: research['hybrid'][k] for k in ['max_depth', 'n_estimators']}, hybrid_cfg)
        hybrid.fit(residuals[FEATURES].to_numpy() * cfg['scale'], residuals.residual.to_numpy() * cfg['scale'])
        correction = hybrid.predict(block[FEATURES].to_numpy() * cfg['scale']) / cfg['scale']
        garch, garch_log = garch_block(returns, block.index, research['garch']['minimum_history'])
        forecasts = {'HAR rolling': predict(har, block, cfg),
                     'XGBoost rolling': predict(xgb, block, cfg),
                     'HAR + correction': np.maximum(reference_har.loc[block.index].to_numpy() + correction,
                                                     cfg['forecast_floor']),
                     'GARCH-t session': np.maximum(garch, cfg['forecast_floor'])}
        fit_logs.append({'fit_origin': str(origin.date()),
                         'rolling_training_count': len(rolling),
                         'rolling_training_start': str(rolling.index.min().date()),
                         'rolling_training_target_max': str(rolling.target_end.max().date()),
                         'residual_count': len(residuals),
                         'residual_target_max': str(residuals.target_end.max().date()),
                         'xgb_parameters': original_fit['xgb_parameters'], **garch_log})
        for name, values in forecasts.items():
            for (date, row), forecast in zip(block.iterrows(), values):
                rows.append({'origin': date, 'target_end': row.target_end, 'model': name,
                             'fit_origin': origin, 'forecast': float(forecast), 'observed': float(row.target)})
        if progress and origin.month in [1, 7]:
            print(f'{origin.date()}: rolling + residual correction + GARCH complete', flush=True)
    return pd.DataFrame(rows), fit_logs, oof
