"""Monthly expanding fits and nested, chronological XGBoost selection."""
import json
import platform
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from .data import ROOT, audit_and_load, config, make_frame

FEATURES = ['daily', 'weekly', 'monthly']


def training_at(frame, origin):
    return frame.loc[(frame.index < origin) & (frame.target_end <= origin)]


def loss(observed, forecast):
    """Nonnegative QLIKE normalisation; same paired differences as raw QLIKE."""
    ratio = np.asarray(observed) / np.asarray(forecast)
    if np.any(ratio <= 0) or not np.all(np.isfinite(ratio)):
        raise ValueError('This normalisation requires positive, finite variances')
    return ratio - np.log(ratio) - 1


def xgb_model(candidate, cfg):
    return XGBRegressor(**candidate, learning_rate=cfg['learning_rate'],
                        objective='reg:squarederror', tree_method='hist',
                        random_state=cfg['seed'], n_jobs=1,
                        subsample=1.0, colsample_bytree=1.0,
                        reg_lambda=1.0, min_child_weight=10)


def predict(model, frame, cfg):
    raw = model.predict(frame[FEATURES].to_numpy() * cfg['scale']) / cfg['scale']
    return np.maximum(raw, cfg['forecast_floor'])


def tune(frame, outer_year, cfg):
    records = []
    for candidate in cfg['xgb_candidates']:
        fold_losses = []
        for year in (outer_year - 2, outer_year - 1):
            validation = frame.loc[frame.index.year == year]
            # Only matured outcomes before the outer year may inform selection.
            validation = validation.loc[validation.target_end < pd.Timestamp(outer_year, 1, 1)]
            if validation.empty:
                raise ValueError('Empty chronological validation fold')
            fit_origin = validation.index[0]
            training = training_at(frame, fit_origin)
            if len(training) < 500:
                raise ValueError('Insufficient inner history')
            model = xgb_model(candidate, cfg)
            model.fit(training[FEATURES].to_numpy() * cfg['scale'],
                      training.target.to_numpy() * cfg['scale'])
            fold_losses.extend(loss(validation.target, predict(model, validation, cfg)))
            records.append({'outer_year': outer_year, 'inner_year': year,
                            **candidate, 'training_count': len(training),
                            'fit_origin': str(fit_origin.date()),
                            'latest_training_target': str(training.target_end.max().date()),
                            'validation_target_max': str(validation.target_end.max().date()),
                            'validation_count': len(validation),
                            'mean_qlike': float(np.mean(fold_losses[-len(validation):]))})
        candidate_score = float(np.mean(fold_losses))
        if 'best_score' not in locals() or candidate_score < best_score:
            best_score, best = candidate_score, candidate
    return dict(best), records


def run(frame, cfg, progress=True):
    evaluation = frame.loc[cfg['development_start']:cfg['development_end']]
    if evaluation.empty:
        raise ValueError('No development observations')
    if evaluation.target_end.max() >= pd.Timestamp(cfg['holdout_start']):
        raise ValueError('Development outcomes cross the locked holdout boundary')
    predictions, tuning, fits = [], [], []
    for year in sorted(set(evaluation.index.year)):
        # Passing only matured past rows also prevents accidental future use in tuning.
        historical = frame.loc[frame.target_end < pd.Timestamp(year, 1, 1)]
        candidate, trials = tune(historical, year, cfg)
        tuning.extend(trials)
        year_rows = evaluation.loc[evaluation.index.year == year]
        if progress:
            print(f'{year}: selected {candidate}; {len(year_rows)} forecasts per model', flush=True)
        for _, block in year_rows.groupby(year_rows.index.to_period('M')):
            origin = block.index[0]
            training = training_at(frame, origin)
            started = time.perf_counter()
            x, y = training[FEATURES].to_numpy() * cfg['scale'], training.target.to_numpy() * cfg['scale']
            har = LinearRegression().fit(x, y)
            xgb = xgb_model(candidate, cfg).fit(x, y)
            forecasts = {'Persistence': block.daily.to_numpy(), 'EWMA': block.ewma.to_numpy(),
                         'HAR': predict(har, block, cfg), 'XGBoost': predict(xgb, block, cfg)}
            fits.append({'fit_origin': str(origin.date()), 'training_count': len(training),
                         'latest_training_target': str(training.target_end.max().date()),
                         'seconds': time.perf_counter() - started, 'xgb_parameters': candidate,
                         'har_intercept_scaled': float(har.intercept_),
                         'har_coefficients': har.coef_.tolist()})
            for name, values in forecasts.items():
                for (date, row), forecast in zip(block.iterrows(), values):
                    predictions.append({'origin': date, 'target_end': row.target_end,
                                        'model': name, 'fit_origin': origin,
                                        'training_target_max': training.target_end.max(),
                                        'forecast': max(float(forecast), cfg['forecast_floor']),
                                        'observed': float(row.target)})
    return pd.DataFrame(predictions), tuning, fits


def main():
    cfg = config()
    series, audit = audit_and_load(cfg)
    frame = make_frame(series, cfg['ewma_decay'])
    predictions, tuning, fits = run(frame, cfg)
    out = ROOT / 'artifacts/development'
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / 'predictions.csv', index=False)
    (out / 'tuning.json').write_text(json.dumps(tuning, indent=2), encoding='utf-8')
    (out / 'fits.json').write_text(json.dumps(fits, indent=2), encoding='utf-8')
    metadata = {'config': cfg, 'python': platform.python_version(),
                'packages': {name: version(name) for name in
                             ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'matplotlib', 'exchange-calendars']},
                'eligible_examples': len(frame), 'forecast_rows': len(predictions),
                'holdout_evaluated': False}
    (out / 'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    from .report import render
    render(predictions, metadata, audit)


if __name__ == '__main__':
    main()
