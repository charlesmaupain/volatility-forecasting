"""Frozen neural study with exact prior-benchmark alignment and seed diagnostics."""
import hashlib
import json
import time
from importlib.metadata import version
import numpy as np
import pandas as pd
from .data import ROOT, config
from .experiment import training_at
from .horizons import horizon_frame, fingerprint_previous
from .inference import paired_losses, bootstrap_intervals, causal_regimes
from .research import load_development
from .neural import neural_forecast, glm_forecast


def run_neural(frame, horizon, cfg, protocol, progress=True):
    evaluation = frame.loc[cfg['development_start']:cfg['development_end']]
    if evaluation.empty or evaluation.target_end.max() >= pd.Timestamp(cfg['holdout_start']):
        raise ValueError('Empty evaluation or holdout boundary crossed')
    rows, fits = [], []
    for month, block in evaluation.groupby(evaluation.index.to_period('M')):
        origin = block.index[0]
        training = training_at(frame, origin)
        if len(training) < 500:
            raise ValueError('Insufficient training history')
        common = dict(horizon=horizon, fit_origin=origin, training_start=training.index.min(),
                      training_target_max=training.target_end.max(), training_count=len(training))
        values, info = glm_forecast(training, block, protocol)
        forecasts = {'Gamma GLM': values}
        fits.append(dict(common, model='Gamma GLM', **info))
        seeds = []
        for seed in protocol['seeds']:
            values, info = neural_forecast(training, block, protocol, seed)
            name = f'MLP seed {seed}'
            forecasts[name] = values
            seeds.append(values)
            fits.append(dict(common, model=name, **info))
        forecasts['MLP ensemble'] = np.mean(seeds, axis=0)
        for name, values in forecasts.items():
            for (date, row), value in zip(block.iterrows(), values):
                rows.append(dict(horizon=horizon, origin=date, target_end=row.target_end,
                                 fit_origin=origin, training_target_max=training.target_end.max(),
                                 model=name, observed=float(row.target), forecast=float(value)))
        if progress and month.month in (1, 7):
            print(f'h={horizon} {month}: {len(training)} matured training rows', flush=True)
    return pd.DataFrame(rows), fits


def align_reference(reference, new):
    expected = new[new.model == 'Gamma GLM'].set_index('origin').sort_index()
    for _, group in reference.groupby('model'):
        current = group.set_index('origin').sort_index()
        pd.testing.assert_index_equal(current.index.as_unit('ns'), expected.index.as_unit('ns'))
        np.testing.assert_array_equal(current.target_end.to_numpy(dtype='datetime64[ns]'),
                                      expected.target_end.to_numpy(dtype='datetime64[ns]'))
        np.testing.assert_allclose(current.observed, expected.observed, rtol=1e-12, atol=0)
    if set(reference.model) != {'HAR', 'XGBoost Gamma'}:
        raise ValueError('Missing reference models')


def main():
    started = time.perf_counter()
    cfg = config()
    protocol = json.loads((ROOT/'configs/neural.json').read_text())
    source = ROOT/'artifacts/research_v3/predictions.csv'
    if not source.exists():
        raise FileNotFoundError('Run python -m volatility.horizons first')
    previous = fingerprint_previous()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    reference = pd.read_csv(source, parse_dates=['origin','target_end','fit_origin','training_target_max'])
    series, _ = load_development(cfg)
    predictions, all_fits, intervals, metrics, annual, regimes = [], [], [], [], [], []
    for horizon in protocol['horizons']:
        new, fits = run_neural(horizon_frame(series, horizon, cfg['ewma_decay']), horizon, cfg, protocol)
        old = reference[(reference.horizon == horizon) & reference.model.isin(['HAR','XGBoost Gamma'])]
        align_reference(old, new)
        combined = pd.concat([old, new], ignore_index=True)
        primary = combined[~combined.model.str.startswith('MLP seed')]
        differences, _ = paired_losses(primary)
        _, losses = paired_losses(combined)
        losses['squared_error'] = (losses.observed-losses.forecast)**2
        metric = losses.groupby('model').agg(qlike=('qlike','mean'), mse=('squared_error','mean'), observations=('qlike','size')).reset_index()
        metrics.append(metric.assign(horizon=horizon))
        annual.append(losses.assign(year=losses.origin.dt.year).groupby(['year','model']).qlike.mean().reset_index().assign(horizon=horizon))
        regime = causal_regimes(series, differences.index)
        regimes.append(losses.merge(regime[['regime']], left_on='origin', right_index=True).groupby(['regime','model']).qlike.agg(['mean','size']).reset_index().assign(horizon=horizon))
        contrast, _ = paired_losses(primary[primary.model.isin(['Gamma GLM','MLP ensemble'])], reference='Gamma GLM')
        for length in protocol['bootstrap_block_lengths']:
            for reference_name, diff in [('HAR', differences), ('Gamma GLM', contrast)]:
                result = bootstrap_intervals(diff, length, protocol['bootstrap_repetitions'], protocol['bootstrap_seed'], protocol['confidence'])
                intervals.append(result.assign(horizon=horizon, reference=reference_name))
        predictions.append(combined)
        all_fits.extend(fits)
        print(metric.sort_values('qlike').to_string(index=False), flush=True)
    artifacts, reports = ROOT/'artifacts/research_v4', ROOT/'reports/research_v4'
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    unchanged = previous == fingerprint_previous() and source_hash == hashlib.sha256(source.read_bytes()).hexdigest()
    if not unchanged:
        raise RuntimeError('Prior outputs changed')
    pd.concat(predictions).to_csv(artifacts/'predictions.csv', index=False)
    (artifacts/'fits.json').write_text(json.dumps(all_fits, indent=2, default=str))
    for name, parts in [('metrics',metrics),('uncertainty',intervals),('annual_metrics',annual),('regime_metrics',regimes)]:
        pd.concat(parts).to_csv(reports/f'{name}.csv', index=False)
    fit_table = pd.DataFrame(all_fits)
    fit_table.groupby(['horizon','model']).agg(fits=('seconds','size'), total_fit_seconds=('seconds','sum'), parameters=('parameters','max')).to_csv(reports/'compute.csv')
    metadata = dict(protocol=protocol, base_config=cfg, prior_v3_sha256=source_hash, previous_outputs_unchanged=unchanged,
                    holdout_evaluated=False, elapsed_seconds=time.perf_counter()-started,
                    packages={p:version(p) for p in ['torch','numpy','pandas','scikit-learn']})
    (reports/'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    from .neural_report import render
    render(pd.concat(metrics), pd.concat(intervals), pd.concat(annual), metadata)


if __name__ == '__main__':
    main()
