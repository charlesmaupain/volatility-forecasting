"""Separate figures and report for the exploratory second experiment."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .data import ROOT


def markdown_table(frame, number_format='.4f'):
    columns = [str(c) for c in frame.columns]
    header = '| ' + ' | '.join([str(frame.index.name or 'Model')] + columns) + ' |\n'
    header += '| ' + ' | '.join(['---'] + ['---:'] * len(columns)) + ' |\n'
    for index, row in frame.iterrows():
        values = [format(float(v), number_format) if isinstance(v, (int, float, np.number)) else str(v) for v in row]
        header += '| ' + ' | '.join([str(index)] + values) + ' |\n'
    return header


def render(metrics, intervals, annual, regimes, ranges, metadata):
    folder = ROOT / 'reports/research_v2'
    figures = folder / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'text.color': '#172c45',
                         'axes.edgecolor': '#ced5de', 'axes.labelcolor': '#354256'})
    def save(fig, name):
        fig.savefig(figures / f'{name}.png', dpi=180, bbox_inches='tight')
        fig.savefig(figures / f'{name}.svg', bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    colors = ['#126b80' if 'HAR' in m else '#8652ad' if 'XGBoost' in m else '#9ba6b2' for m in metrics.index]
    ax.scatter(metrics.qlike, np.arange(len(metrics)), c=colors, s=65, zorder=3)
    ax.set_yticks(np.arange(len(metrics)), metrics.index)
    for i, value in enumerate(metrics.qlike):
        ax.annotate(f'{value:.3f}', (value, i), xytext=(8, 0), textcoords='offset points', va='center')
    ax.invert_yaxis()
    ax.set_title('Research extensions on the same development dates', loc='left')
    if metrics.qlike.max() > 10 * metrics.qlike.min():
        ax.set_xscale('log')
        ax.set_xlim(metrics.qlike.min() * .8, metrics.qlike.max() * 2.3)
        ax.set_xlabel('Mean normalised QLIKE (log scale) | lower is better')
    else:
        ax.set_xlabel('Mean normalised QLIKE | lower is better')
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    save(fig, 'extended_comparison')
    pair = intervals[intervals.model == 'XGBoost'].sort_values('block_length')
    fig, ax = plt.subplots(figsize=(10, 3.7), layout='constrained')
    for position, (_, row) in enumerate(pair.iterrows()):
        ax.plot([row.pointwise_low, row.pointwise_high], [position, position], color='#8652ad', lw=3)
        ax.scatter(row.mean_difference, position, color='#172c45', s=40, zorder=3)
    ax.axvline(0, color='#126b80', ls='--')
    ax.set_yticks(range(len(pair)), [f'{int(b)}-session expected blocks' for b in pair.block_length])
    ax.set_xlabel('Mean QLIKE: original XGBoost minus original HAR | positive favours HAR')
    ax.set_title('Original XGBoost-HAR gap: dependence sensitivity', loc='left', pad=28)
    ax.text(0, 1.02, 'Pointwise 95% stationary-bootstrap intervals | 5,000 replicates | development data',
            transform=ax.transAxes, fontsize=9, color='#617085')
    ax.grid(axis='x', alpha=.15)
    save(fig, 'bootstrap_sensitivity')
    fig, ax = plt.subplots(figsize=(12, 4), layout='constrained')
    for model, color in [('HAR rolling', '#126b80'), ('XGBoost rolling', '#8652ad')]:
        reference = 'HAR' if model.startswith('HAR') else 'XGBoost'
        ax.plot(annual.index, annual[model] - annual[reference], marker='o', color=color, label=model)
    ax.axhline(0, color='#8d98a5', lw=1)
    ax.set_xlabel('Development forecast-origin year')
    ax.set_yscale('symlog', linthresh=.05)
    rolling_deltas = np.concatenate([(annual['HAR rolling'] - annual['HAR']).to_numpy(), (annual['XGBoost rolling'] - annual['XGBoost']).to_numpy()])
    ax.set_ylim(min(-.2, rolling_deltas.min() * 1.5), max(.2, rolling_deltas.max() * 1.5))
    ax.set_yticks([-.1, 0, .1, 1, 100, 10000], ['-0.1', '0', '0.1', '1', '100', '10,000'])
    ax.set_ylabel('Rolling minus expanding QLIKE\nSymmetric log scale; negative = improvement')
    ax.set_title('Does a shorter training history improve adaptation?', loc='left')
    ax.legend(frameon=False)
    ax.grid(alpha=.15)
    save(fig, 'rolling_sensitivity')
    primary = intervals[intervals.block_length == metadata['research_config']['bootstrap_primary_block_length']]
    primary = primary.set_index('model')[['mean_difference', 'pointwise_low', 'pointwise_high', 'family_low', 'family_high']]
    table = metrics.copy()
    table['floor_hits'] = [metadata['floor_hits'][m] for m in table.index]
    table['observations'] = table.observations.astype(str)
    table['floor_hits'] = table.floor_hits.astype(str)
    # Preserve small MSE values without rounding them to zero.
    table['mse'] = table.mse.map(lambda x: f'{x:.3e}')
    best = metrics.index[0]
    note = f'''# Part 2: uncertainty, adaptation and additional benchmarks

**Exploratory development research. The 2018-2019 final holdout remains unevaluated.**

The lowest average QLIKE in this expanded comparison is achieved by **{best}**. Every model predicts the same {metadata['forecast_dates_per_model']:,} dates. Original model forecasts are reused unchanged, with a SHA-256 fingerprint recorded in run_metadata.json.

![Extended comparison](figures/extended_comparison.png)

{markdown_table(table)}

Read [the interpretation](../../docs/findings_v2.md) for the distinction between a small estimated hybrid gain and evidence strong enough to trust it. Rolling XGBoost produced six negative raw variance predictions, retained at the declared floor; this failure dominates its loss and makes the unstudentised family bands uninformative. No failed dates were removed.

QLIKE is dimensionless and lower is better. Variance MSE is in squared-variance units. Floor hits count forecasts clipped to the common positive floor; inspect them before interpreting a model's average loss.

## Uncertainty in paired loss differences

![Bootstrap sensitivity](figures/bootstrap_sensitivity.png)

Primary expected block length: 20 sessions; 5,000 stationary-bootstrap replicates. All seven difference series share sampled dates. The sensitivity table also reports expected block lengths of 5 and 60 in uncertainty.csv.

{markdown_table(primary)}

Positive differences favour expanding HAR. Pointwise intervals apply to individual comparisons; family bands use the bootstrap maximum absolute centred mean deviation across the seven comparisons. Family bands address this finite simultaneous comparison family, not the whole adaptive research process. They may be wide if one model has extreme loss variation.

These intervals are conditional on saved forecast paths. Models are not refitted or reselected within each bootstrap replicate. They rely on approximate stationarity and sufficient short-range dependence representation, which structural breaks can undermine. The choice to investigate these variants followed inspection of v1, so these are not independent confirmatory results. No bootstrap frequency is presented as a p-value.

## Training-window sensitivity

![Rolling sensitivity](figures/rolling_sensitivity.png)

Rolling models use the last 756 eligible training rows. XGBoost retains the hyperparameters selected from past data in the original expanding experiment, isolating the window change rather than tuning a new rolling model. The comparison does not establish the optimal training-window length.

## Volatility regimes known at forecast time

Regimes use the current five-session RV average compared with expanding 25th/75th percentile thresholds calculated from prior weekly averages only. All models use the same regime assignments. They do not use future realised volatility to define a forecast-time regime.

{markdown_table(regimes.qlike.unstack().reindex(['Low', 'Middle', 'High']))}

## Extrapolation diagnostic

A row is outside the original monthly training range if any of its three HAR/XGBoost features is below the historical minimum or above the maximum. This diagnostic uses information available at the fit and forecast times, not outcomes. It is descriptive and does not establish why a model wins or loses.

{markdown_table(ranges.qlike.unstack())}

See regime_metrics.csv and range_metrics.csv for sample sizes; annual_metrics.csv retains every development year. The 2008 and 2017 periods were selected after v1 and are descriptive case studies.

## Out-of-sample HAR correction

The correction model learns actual variance minus historical HAR forecasts generated by earlier monthly fits. A residual enters training only after its target session has ended. The correction model has fixed depth 2, 100 trees and learning rate 0.03, and uses the same three variance features. Minimum matured residual count in this run: {metadata['minimum_matured_residual_count']}.

The correction is additive in variance levels, with the common positivity floor. It can overcorrect in low-volatility periods; a failed correction is retained rather than retrospectively replaced with a better-looking specification.

## GARCH target and fitting assumptions

GARCH(1,1) uses Student-t innovations and zero conditional mean on open-to-close log returns, scaled to percent. Its next-session return-variance forecast is evaluated against intraday realised variance, a noisy proxy under the stated modelling assumptions. It uses returns rather than the matched variance features, so it is an additional session-risk comparator.

Missing return sessions are not compressed. Each fit starts after the latest missing return and uses a contiguous history. Parameters are refitted monthly, with variance state updated after each observed session. Non-converged fits: {metadata['garch_nonconverged_fits']}. Maximum fitted alpha + beta: {metadata['maximum_garch_persistence']:.6f}. Boundary persistence near one warrants caution about long-run variance interpretations.

## Reproduce and inspect

```bash
python -m unittest discover -s tests -v
python -m volatility.research
```

Run the original experiment first if artifacts/development is absent. New detailed forecasts, historical out-of-sample HAR residual sources and fit lineage are in artifacts/research_v2 (Git-ignored). This report and aggregate diagnostics are separate from v1.

Methods: [stationary bootstrap documentation](https://arch.readthedocs.io/en/latest/bootstrap/timeseries-bootstraps.html), [GARCH forecast recursion](https://arch.readthedocs.io/en/latest/univariate/forecasting.html).

Multi-instrument robustness, five-day targets and feature-group ablations remain Part 2 work. Economic backtesting and final holdout evaluation remain Part 3 work.
'''
    (folder / 'research.md').write_text(note, encoding='utf-8')


