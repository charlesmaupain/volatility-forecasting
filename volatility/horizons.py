"""Direct-horizon variance forecasts with an explicitly positive objective."""
import hashlib
import json
from importlib.metadata import version
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .data import ROOT, config, make_frame
from .experiment import FEATURES, training_at, loss, xgb_model
from .inference import paired_losses, bootstrap_intervals, floor_mask
from .research import load_development


MODEL_OBJECTIVES = {'XGBoost squared': 'reg:squarederror', 'XGBoost Gamma': 'reg:gamma'}


def horizon_frame(series, horizon, decay=.94):
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    observed = series.dropna()
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError('Observed variances must be positive and finite')
    frame = make_frame(series, decay)
    future = pd.concat([series.shift(-step) for step in range(1, horizon + 1)], axis=1)
    frame['target'] = future.sum(axis=1, min_count=horizon)
    frame['target_end'] = pd.Series(series.index, index=series.index).shift(-horizon)
    frame['ewma'] *= horizon
    return frame.dropna()


def model(candidate, objective, cfg):
    if objective not in MODEL_OBJECTIVES.values():
        raise ValueError('Unsupported objective')
    return xgb_model(candidate, cfg).set_params(objective=objective)


def forecast(fitted, rows, cfg, positive=False):
    raw = np.asarray(fitted.predict(rows[FEATURES].to_numpy() * cfg['scale']), dtype=float) / cfg['scale']
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite forecast')
    if positive and (raw <= 0).any():
        raise ValueError('Gamma failed positive-output requirement')
    # Do not use a floor to conceal Gamma underflow or nonpositive outputs.
    return raw if positive else np.maximum(raw, cfg['forecast_floor']), int((raw <= 0).sum())


def tune_objective(frame, year, objective, cfg, minimum=500):
    best_score, best = float('inf'), None
    records = []
    for candidate in cfg['xgb_candidates']:
        scores = []
        for inner_year in (year - 2, year - 1):
            validation = frame.loc[(frame.index.year == inner_year) & (frame.target_end < pd.Timestamp(year, 1, 1))]
            if validation.empty:
                raise ValueError('Empty chronological validation fold')
            origin = validation.index[0]
            training = training_at(frame, origin)
            if len(training) < minimum:
                raise ValueError('Insufficient inner training history')
            fitted = model(candidate, objective, cfg).fit(training[FEATURES].to_numpy() * cfg['scale'],
                                                         training.target.to_numpy() * cfg['scale'])
            values, failures = forecast(fitted, validation, cfg, positive=objective == 'reg:gamma')
            current = loss(validation.target, values)
            scores.extend(current)
            records.append({'outer_year': year, 'inner_year': inner_year, 'objective': objective,
                            **candidate, 'fit_origin': str(origin.date()),
                            'training_target_max': str(training.target_end.max().date()),
                            'validation_target_max': str(validation.target_end.max().date()),
                            'training_count': len(training), 'validation_count': len(validation),
                            'mean_qlike': float(current.mean()), 'raw_nonpositive': failures})
        score = float(np.mean(scores))
        if score < best_score:
            best_score, best = score, dict(candidate)
    if best is None:
        raise ValueError('No selected candidate')
    return best, records


def run_horizon(frame, horizon, cfg, protocol, progress=True):
    evaluation = frame.loc[cfg['development_start']:cfg['development_end']]
    if evaluation.empty or evaluation.target_end.max() >= pd.Timestamp(cfg['holdout_start']):
        raise ValueError('Empty evaluation or final holdout boundary crossed')
    rows, trials, fits = [], [], []
    for year in sorted(set(evaluation.index.year)):
        known = frame.loc[frame.target_end < pd.Timestamp(year, 1, 1)]
        selections = {}
        for name, objective in MODEL_OBJECTIVES.items():
            candidate, logs = tune_objective(known, year, objective, cfg)
            selections[name] = candidate
            trials.extend([dict(log, horizon=horizon) for log in logs])
        annual = evaluation.loc[evaluation.index.year == year]
        if progress:
            print(f'h={horizon}, {year}: {selections}', flush=True)
        for _, block in annual.groupby(annual.index.to_period('M')):
            origin = block.index[0]
            training = training_at(frame, origin)
            rolling = training.tail(protocol['rolling_observations'])
            if len(rolling) != protocol['rolling_observations']:
                raise ValueError('Insufficient rolling history')
            har = LinearRegression().fit(training[FEATURES].to_numpy() * cfg['scale'],
                                         training.target.to_numpy() * cfg['scale'])
            har_values, har_failures = forecast(har, block, cfg)
            forecasts = {'Persistence': (block.daily.to_numpy() * horizon, 0, training),
                         'EWMA': (block.ewma.to_numpy(), 0, training),
                         'HAR': (har_values, har_failures, training)}
            for name, objective in MODEL_OBJECTIVES.items():
                fitted = model(selections[name], objective, cfg).fit(
                    training[FEATURES].to_numpy() * cfg['scale'], training.target.to_numpy() * cfg['scale'])
                values, failures = forecast(fitted, block, cfg, positive=objective == 'reg:gamma')
                forecasts[name] = (values, failures, training)
            fitted = model(selections['XGBoost Gamma'], 'reg:gamma', cfg).fit(
                rolling[FEATURES].to_numpy() * cfg['scale'], rolling.target.to_numpy() * cfg['scale'])
            values, failures = forecast(fitted, block, cfg, positive=True)
            forecasts['XGBoost Gamma rolling'] = (values, failures, rolling)
            for name, (values, failures, used) in forecasts.items():
                fits.append({'horizon': horizon, 'fit_origin': str(origin.date()), 'model': name,
                             'training_start': str(used.index.min().date()), 'training_count': len(used),
                             'training_target_max': str(used.target_end.max().date()),
                             'raw_nonpositive': failures, 'floor_hits': int(floor_mask(values, cfg['forecast_floor']).sum()),
                             'selected_parameters': selections.get(name.replace(' rolling', ''), {})})
                for (date, row), value in zip(block.iterrows(), values):
                    rows.append({'horizon': horizon, 'origin': date, 'target_end': row.target_end,
                                 'fit_origin': origin, 'training_target_max': used.target_end.max(),
                                 'model': name, 'forecast': float(value), 'observed': float(row.target)})
    return pd.DataFrame(rows), trials, fits


def fingerprint_previous():
    files = ['artifacts/development/predictions.csv', 'artifacts/development/fits.json',
             'artifacts/research_v2/predictions.csv', 'artifacts/research_v2/fits.json']
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in files if (ROOT / name).exists()}


def main():
    cfg = config()
    protocol = json.loads((ROOT / 'configs/positive_horizons.json').read_text(encoding='utf-8-sig'))
    previous = fingerprint_previous()
    series, _ = load_development(cfg)
    all_predictions, all_trials, all_fits, interval_parts, metric_parts = [], [], [], [], []
    for horizon in protocol['horizons']:
        frame = horizon_frame(series, horizon, cfg['ewma_decay'])
        predictions, trials, fits = run_horizon(frame, horizon, cfg, protocol)
        differences, losses = paired_losses(predictions)
        for block in protocol['bootstrap_block_lengths']:
            interval = bootstrap_intervals(differences, block, protocol['bootstrap_repetitions'],
                                           protocol['seed'], protocol['confidence'])
            interval['horizon'] = horizon
            interval_parts.append(interval)
        metrics = losses.assign(squared_error=(losses.observed-losses.forecast)**2).groupby('model').agg(
            qlike=('qlike', 'mean'), mse=('squared_error', 'mean'), observations=('qlike', 'size')).reset_index()
        metrics['horizon'] = horizon
        metric_parts.append(metrics)
        all_predictions.append(predictions)
        all_trials.extend(trials)
        all_fits.extend(fits)
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(metric_parts, ignore_index=True)
    intervals = pd.concat(interval_parts, ignore_index=True)
    old = ROOT / 'artifacts/development/predictions.csv'
    reproduced = False
    if old.exists():
        original = pd.read_csv(old, parse_dates=['origin', 'target_end'])
        for name in ['HAR', 'EWMA', 'Persistence', 'XGBoost']:
            current_name = 'XGBoost squared' if name == 'XGBoost' else name
            current = predictions[(predictions.horizon == 1) & (predictions.model == current_name)].set_index('origin').sort_index()
            reference = original[original.model == name].set_index('origin').sort_index()
            pd.testing.assert_index_equal(current.index.as_unit('ns'), reference.index.as_unit('ns'))
            np.testing.assert_allclose(current.forecast, reference.forecast, rtol=1e-6, atol=1e-12)
            np.testing.assert_allclose(current.observed, reference.observed, rtol=1e-12)
        reproduced = True
    artifacts = ROOT / 'artifacts/research_v3'
    reports = ROOT / 'reports/research_v3'
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(artifacts / 'predictions.csv', index=False)
    (artifacts / 'tuning.json').write_text(json.dumps(all_trials, indent=2), encoding='utf-8')
    (artifacts / 'fits.json').write_text(json.dumps(all_fits, indent=2), encoding='utf-8')
    metrics.to_csv(reports / 'metrics.csv', index=False)
    intervals.to_csv(reports / 'uncertainty.csv', index=False)
    fit_frame = pd.DataFrame(all_fits)
    diagnostics = fit_frame.groupby(['horizon', 'model'])[['raw_nonpositive', 'floor_hits']].sum()
    diagnostics.to_csv(reports / 'positivity_audit.csv')
    metadata = {'protocol': protocol, 'base_config': cfg, 'previous_file_hashes': previous,
                'previous_outputs_unchanged': fingerprint_previous() == previous,
                'original_one_session_predictions_reproduced': reproduced,
                'holdout_evaluated': False,
                'packages': {p: version(p) for p in ['numpy', 'pandas', 'xgboost', 'scikit-learn', 'arch']}}
    if not metadata['previous_outputs_unchanged']:
        raise RuntimeError('Previous outputs changed')
    (reports / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    from .horizon_report import render
    render(metrics, intervals, diagnostics, metadata)
    print(metrics.sort_values(['horizon', 'qlike']).to_string(index=False), flush=True)
    print(diagnostics.to_string(), flush=True)


if __name__ == '__main__':
    main()

