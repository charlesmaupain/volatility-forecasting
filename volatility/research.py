"""Part 2 orchestration; all new outputs are separate from the v1 benchmark."""
import hashlib
import json
from importlib.metadata import version
import numpy as np
import pandas as pd
from .data import ROOT, config, acquire, session_dates, make_frame
from .experiment import FEATURES, training_at
from .inference import paired_losses, bootstrap_intervals, causal_regimes, floor_mask
from .research_models import extension_predictions


def load_development(cfg):
    import exchange_calendars as xc
    raw = pd.read_csv(acquire(cfg))
    selected = raw.loc[raw.Symbol == cfg['symbol']].copy()
    selected['date'] = session_dates(selected.iloc[:, 0])
    selected = selected.set_index('date').sort_index().loc[cfg['history_start']:cfg['development_end']]
    if selected.index.has_duplicates:
        raise ValueError('Duplicate instrument dates')
    expected = xc.get_calendar(cfg['calendar'], start='1999-01-01', end='2021-01-01').sessions_in_range(
        cfg['history_start'], cfg['development_end']).tz_localize(None)
    if len(selected.index.difference(expected)):
        raise ValueError('Unexpected session dates')
    columns = [cfg['measure'], 'open_to_close', 'open_price', 'close_price']
    if not np.isfinite(selected[columns].to_numpy()).all() or (selected[[cfg['measure'], 'open_price', 'close_price']] <= 0).any().any():
        raise ValueError('Invalid source values')
    np.testing.assert_allclose(selected.open_to_close,
                               np.log(selected.close_price / selected.open_price), atol=1e-12, rtol=1e-10)
    return selected[cfg['measure']].reindex(expected), selected.open_to_close.reindex(expected)


def main():
    cfg = config()
    research = json.loads((ROOT / 'configs/research.json').read_text(encoding='utf-8-sig'))
    original = ROOT / 'artifacts/development/predictions.csv'
    if not original.exists():
        raise FileNotFoundError('Run python -m volatility.experiment first')
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    original_fits_path = ROOT / 'artifacts/development/fits.json'
    fits_hash = hashlib.sha256(original_fits_path.read_bytes()).hexdigest()
    baseline = pd.read_csv(original, parse_dates=['origin', 'target_end', 'fit_origin', 'training_target_max'])
    if baseline.target_end.max() >= pd.Timestamp(cfg['holdout_start']):
        raise ValueError('Original outputs contain holdout outcomes')
    original_run = json.loads((ROOT / 'artifacts/development/run.json').read_text())
    if original_run['config'] != cfg:
        raise ValueError('Current config differs from the original benchmark')
    original_fits = json.loads(original_fits_path.read_text())
    series, returns = load_development(cfg)
    frame = make_frame(series, cfg['ewma_decay'])
    # Prevent combining changed data with stale original outputs.
    full_origins = frame.loc[cfg['development_start']:cfg['development_end']].index
    if not pd.DatetimeIndex(sorted(baseline.origin.unique())).equals(full_origins):
        raise ValueError('Original benchmark does not cover all expected origins')
    expected = frame.loc[baseline.origin.unique()]
    for _, group in baseline.groupby('model'):
        np.testing.assert_allclose(group.set_index('origin').loc[expected.index].observed, expected.target, rtol=1e-12)
    extra, fits, oof = extension_predictions(frame, returns, baseline, original_fits, cfg, research)
    combined = pd.concat([baseline, extra], ignore_index=True)
    differences, losses = paired_losses(combined)
    intervals = pd.concat([bootstrap_intervals(differences, block, research['bootstrap_repetitions'],
                                                research['seed'], research['confidence'])
                           for block in research['bootstrap_block_lengths']], ignore_index=True)
    origins = differences.index
    regimes = causal_regimes(series, origins, research['regime_quantiles'])
    out_of_range = pd.Series(False, index=origins, name='outside_training_range')
    for fit in original_fits:
        origin = pd.Timestamp(fit['fit_origin'])
        training = training_at(frame, origin)
        block = frame.loc[origins[origins.to_period('M') == origin.to_period('M')], FEATURES]
        out_of_range.loc[block.index] = ((block < training[FEATURES].min()) | (block > training[FEATURES].max())).any(axis=1)
    diagnostics = regimes.join(out_of_range)
    enriched = losses.merge(diagnostics, left_on='origin', right_index=True, validate='many_to_one')
    artifacts = ROOT / 'artifacts/research_v2'
    reports = ROOT / 'reports/research_v2'
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    combined.to_csv(artifacts / 'predictions.csv', index=False)
    oof.to_csv(artifacts / 'har_oof.csv')
    diagnostics.to_csv(artifacts / 'diagnostics.csv')
    (artifacts / 'fits.json').write_text(json.dumps(fits, indent=2), encoding='utf-8')
    intervals.to_csv(reports / 'uncertainty.csv', index=False)
    metrics = losses.assign(squared_error=(losses.observed - losses.forecast) ** 2).groupby('model').agg(
        qlike=('qlike', 'mean'), mse=('squared_error', 'mean'), observations=('qlike', 'size')).sort_values('qlike')
    metrics.to_csv(reports / 'metrics.csv')
    regimes_table = enriched.groupby(['regime', 'model']).agg(qlike=('qlike', 'mean'), observations=('qlike', 'size'))
    regimes_table.to_csv(reports / 'regime_metrics.csv')
    range_table = enriched.groupby(['outside_training_range', 'model']).agg(qlike=('qlike', 'mean'), observations=('qlike', 'size'))
    range_table.to_csv(reports / 'range_metrics.csv')
    annual = losses.assign(year=losses.origin.dt.year).groupby(['year', 'model']).qlike.mean().unstack()
    annual.to_csv(reports / 'annual_metrics.csv')
    metadata = {'original_forecast_sha256': original_hash, 'original_fits_sha256': fits_hash,
                'source_sha256': cfg['source_sha256'], 'research_config': research,
                'base_config': cfg, 'holdout_evaluated': False,
                'models': list(metrics.index),
                'floor_hits': combined.assign(at_floor=floor_mask(combined.forecast, cfg['forecast_floor'])).groupby('model').at_floor.sum().astype(int).to_dict(), 'forecast_dates_per_model': len(origins),
                'packages': {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'arch']},
                'maximum_garch_persistence': max(f['persistence'] for f in fits),
                'garch_nonconverged_fits': sum(f['convergence_flag'] != 0 for f in fits),
                'minimum_matured_residual_count': min(f['residual_count'] for f in fits)}
    (reports / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    assert hashlib.sha256(original.read_bytes()).hexdigest() == original_hash
    assert hashlib.sha256(original_fits_path.read_bytes()).hexdigest() == fits_hash
    from .research_report import render
    render(metrics, intervals, annual, regimes_table, range_table, metadata)
    print(metrics.to_string(), flush=True)
    print(intervals[intervals.block_length == research['bootstrap_primary_block_length']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()


