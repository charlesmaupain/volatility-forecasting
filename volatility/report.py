"""Static, reproducible research figures and a development results note."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import ROOT
from .experiment import loss

COLORS = {'Persistence': '#8c98a6', 'EWMA': '#b8843b', 'HAR': '#126b80', 'XGBoost': '#8652ad'}


def render(predictions, metadata, audit):
    out = ROOT / 'reports'
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    data = predictions.copy()
    data['qlike'] = loss(data.observed, data.forecast)
    data['squared_error'] = (data.observed - data.forecast) ** 2
    metrics = data.groupby('model').agg(qlike=('qlike', 'mean'), mse=('squared_error', 'mean'),
                                      observations=('qlike', 'size')).sort_values('qlike')
    baseline = metrics.loc['HAR', 'qlike']
    metrics['qlike_minus_har'] = metrics.qlike - baseline
    metrics.to_csv(out / 'metrics.csv')
    annual = data.assign(year=data.origin.dt.year).groupby(['year', 'model']).qlike.mean().unstack()
    annual.to_csv(out / 'annual_metrics.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.labelcolor': '#354256',
                         'text.color': '#172c45', 'axes.edgecolor': '#ced5de',
                         'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff'})
    def save(fig, name):
        fig.savefig(figures / (name + '.png'), dpi=180, bbox_inches='tight')
        fig.savefig(figures / (name + '.svg'), bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.4), layout='constrained')
    bars = ax.barh(metrics.index, metrics.qlike, color=[COLORS[m] for m in metrics.index], height=.6)
    ax.invert_yaxis()
    ax.bar_label(bars, fmt='%.3f', padding=7)
    ax.set_xlim(0, metrics.qlike.max() * 1.2)
    ax.set_xlabel('Mean normalised QLIKE | lower is better')
    ax.set_title('Classical structure vs nonlinear prediction', loc='left', pad=25)
    ax.text(0, 1.025, 'S&P 500 intraday variance | 2006-2017 development evaluation | matched forecast dates',
            transform=ax.transAxes, fontsize=9, color='#617085')
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    save(fig, 'model_comparison')

    fig, ax = plt.subplots(figsize=(12, 4.6), layout='constrained')
    base = data[data.model == 'HAR'].set_index('target_end')
    ax.plot(base.index, np.sqrt(base.observed) * 100, color='#aeb8c2', lw=.7, label='Realised')
    for name in ['HAR', 'XGBoost']:
        subset = data[data.model == name].set_index('target_end')
        ax.plot(subset.index, np.sqrt(subset.forecast) * 100, color=COLORS[name], lw=.7,
                alpha=.85, label=name)
    ax.set_ylabel('Session volatility (%) | not annualised')
    ax.set_title('Forecasts and outcomes across market conditions', loc='left')
    ax.legend(frameon=False, ncols=3)
    ax.grid(alpha=.15)
    save(fig, 'forecast_history')

    wide = data.pivot(index='target_end', columns='model', values='qlike')
    fig, ax = plt.subplots(figsize=(12, 4.3), layout='constrained')
    for name in ['Persistence', 'EWMA', 'XGBoost']:
        ax.plot(wide.index, (wide[name] - wide.HAR).cumsum(), color=COLORS[name], label=name, lw=1.6)
    ax.axhline(0, color=COLORS['HAR'], lw=1)
    ax.set_title('Does the forecasting advantage persist?', loc='left')
    ax.set_ylabel('Cumulative QLIKE minus HAR\nPositive = worse than HAR')
    ax.legend(frameon=False, ncols=3)
    ax.grid(alpha=.15)
    save(fig, 'cumulative_loss')

    fig, ax = plt.subplots(figsize=(12, 4.3), layout='constrained')
    annual[['Persistence', 'EWMA', 'HAR', 'XGBoost']].plot.bar(
        ax=ax, color=[COLORS[m] for m in ['Persistence', 'EWMA', 'HAR', 'XGBoost']], width=.8)
    ax.set_title('Performance varies through time', loc='left')
    ax.set_ylabel('Mean normalised QLIKE | lower is better')
    ax.set_xlabel('Development forecast-origin year')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(frameon=False, ncols=4)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    save(fig, 'annual_performance')

    rows = '\n'.join(f'| {name} | {r.qlike:.4f} | {r.mse:.3e} | {r.qlike_minus_har:+.4f} | {int(r.observations):,} |'
                     for name, r in metrics.iterrows())
    best = metrics.index[0]
    note = f'''# Development benchmark: S&P 500 realised variance

**Status: development results; final 2018-2019 holdout remains unevaluated.**

{best} has the lowest average QLIKE in this prespecified first experiment. This is descriptive evidence, not a statistical-significance claim. All four models use identical eligible forecast dates; HAR and XGBoost use identical daily, weekly and monthly variance features.

![Model comparison](figures/model_comparison.png)

| Model | Mean QLIKE | Variance MSE | QLIKE minus HAR | Forecasts |
| --- | ---: | ---: | ---: | ---: |
{rows}

## Design

- Forecast origin: after session close; outcome: next-session intraday variance (rv5).
- Actual origins: {data.origin.min().date()} to {data.origin.max().date()}.
- Actual target dates: {data.target_end.min().date()} to {data.target_end.max().date()}.
- Monthly expanding fits; XGBoost selected annually from four candidates using two preceding chronological validation years.
- EWMA decay 0.94; positive forecast floor 1e-10. HAR and XGBoost fit variance levels, scaled by 10,000.
- QLIKE uses observed/forecast - log(observed/forecast) - 1. Smaller is better; signed differences are directly comparable.
- No trading returns, Sharpe ratio or executable-strategy claims.

![Forecast history](figures/forecast_history.png)

![Cumulative loss](figures/cumulative_loss.png)

![Annual performance](figures/annual_performance.png)

## Data integrity and limitations

Source: Oxford-Man Realized Library archive hosted by the academic highfrequency project; pinned revision `{metadata['config']['source_revision']}`.

There are {audit['rows']:,} instrument observations across the archive and {len(audit['missing_sessions'])} missing XNYS sessions. Civil date labels are preserved; converting midnight labels to UTC would incorrectly move summer dates backwards. Missing sessions are not imputed; incomplete feature/target windows are excluded. Calendar gaps and software versions are recorded in data_audit.json.

This is an S&P 500 index experiment, not an SPY trading backtest. Intraday variance excludes overnight risk. Historical publication timestamps are unavailable, so real-time availability is assumed. The archive ends before the main COVID shock. Regime analysis, dependence-aware uncertainty, expanded tuning/objective experiments and the untouched holdout are future work. Repeated development experiments must be logged; these results are not an independent final test.

## Reproduction

From the repository root, install requirements-lock.txt, then run:

```bash
python -m unittest discover -s tests -v
python -m volatility.experiment
```

The first run retrieves the pinned archive and checks its SHA-256. Raw data, row-level predictions, fit records and tuning logs remain in Git-ignored data/ and artifacts/. Aggregate reports and figures are generated here. The source archive is publicly hosted; a separate data-specific redistribution licence has not been established, so raw data is not bundled.
'''
    (out / 'development.md').write_text(note, encoding='utf-8')
    (out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(metrics.to_string(), flush=True)
