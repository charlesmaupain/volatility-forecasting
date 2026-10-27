"""Publication figures for the positive-output and direct-horizon experiment."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .data import ROOT
from .research_report import markdown_table

COLORS = {'HAR': '#126b80', 'XGBoost squared': '#8652ad', 'XGBoost Gamma': '#c06236',
          'XGBoost Gamma rolling': '#b48635', 'EWMA': '#8995a2', 'Persistence': '#b1bbc4'}


def render(metrics, intervals, diagnostics, metadata):
    folder = ROOT / 'reports/research_v3'
    figures = folder / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'text.color': '#172c45',
                         'axes.edgecolor': '#ced5de', 'axes.labelcolor': '#354256'})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for axis, horizon in zip(axes, [1, 5]):
        subset = metrics[metrics.horizon == horizon].sort_values('qlike')
        axis.barh(subset.model, subset.qlike, color=[COLORS[m] for m in subset.model])
        axis.invert_yaxis()
        axis.set_title(f'{horizon}-session variance target', loc='left')
        axis.set_xlabel('Mean QLIKE | lower is better')
        axis.set_xlim(0, subset.qlike.max() * 1.22)
        axis.bar_label(axis.containers[0], fmt='%.3f', padding=5)
        axis.grid(axis='x', alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle('Positive variance forecasts across two horizons', fontsize=15, weight='bold')
    for suffix in ['png', 'svg']:
        fig.savefig(figures / f'horizon_comparison.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    primary = intervals[intervals.block_length == metadata['protocol']['bootstrap_primary_block_length']]
    models = ['XGBoost squared', 'XGBoost Gamma', 'XGBoost Gamma rolling']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout='constrained')
    for axis, horizon in zip(axes, [1, 5]):
        subset = primary[primary.horizon == horizon].set_index('model').loc[models]
        for position, (name, row) in enumerate(subset.iterrows()):
            axis.plot([row.pointwise_low, row.pointwise_high], [position, position], color=COLORS[name], lw=3)
            axis.scatter(row.mean_difference, position, color=COLORS[name], s=40, zorder=3)
        axis.set_yticks(range(len(models)), models)
        axis.invert_yaxis()
        axis.axvline(0, color='#126b80', ls='--')
        axis.set_title(f'{horizon}-session target', loc='left')
        axis.set_xlabel('QLIKE difference from HAR | negative favours ML')
        axis.grid(axis='x', alpha=.15)
    fig.suptitle('Pointwise 95% stationary-bootstrap intervals | 20-session expected blocks', fontsize=13, weight='bold')
    for suffix in ['png', 'svg']:
        fig.savefig(figures / f'horizon_uncertainty.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    tables = []
    for horizon in [1, 5]:
        sub = metrics[metrics.horizon == horizon].sort_values('qlike').set_index('model').drop(columns='horizon')
        sub['mse'] = sub.mse.map(lambda x: f'{x:.3e}')
        sub['observations'] = sub.observations.astype(str)
        tables.append(f'## {horizon}-session target\n\n' + markdown_table(sub))
    uncertainty_table = primary.set_index(['horizon', 'model'])[['mean_difference', 'pointwise_low', 'pointwise_high', 'family_low', 'family_high']]
    audit_table = diagnostics.astype(str)
    note = '''# Positive-output modelling and five-session forecasts

**Exploratory development results, 2006-2017. The final holdout remains unevaluated.**

Gamma regression with a log link predicts a positive conditional mean. Its fixed-shape mean-dependent loss is proportional to log(mu) + y/mu, aligning with QLIKE. This is a quasi-likelihood choice, not an assertion that variance follows an exact Gamma distribution. Unlike fitting log targets with squared error, it does not need a lognormal mean retransformation.

![Comparison](figures/horizon_comparison.png)

''' + '\n\n'.join(tables) + '''

The five-session target is the sum of the next five intraday realised variances. It is not the variance of the cumulative five-session return. Models use identical dates within a horizon; horizons have different eligible sample counts. The variance MSE values have different target scales, so comparisons across horizons should not be treated as a ranking of forecast difficulty.

## Dependence-aware uncertainty

![Intervals](figures/horizon_uncertainty.png)

''' + markdown_table(uncertainty_table) + '''

Differences are model minus HAR. Negative values favour the listed model. These are 5,000-replicate stationary-bootstrap intervals, with 20-session expected blocks primary and 5/60-session sensitivity in uncertainty.csv. Five-session outcomes overlap; the block procedure accounts for local dependence approximately, not through an assumption that observations are independent.

Pointwise intervals concern individual comparisons. Family bands cover the specified five-comparison family within each horizon approximately; they are not adjusted for both horizons, all earlier experiments or adaptive research choices. Models are not refitted or retuned inside the bootstrap. Structural breaks can violate the approximate-stationarity assumption.

## Positivity audit

''' + markdown_table(audit_table) + '''

Gamma forecasts are checked before any clipping; a nonpositive or nonfinite output stops the run. Squared-error models retain the original declared numerical floor. The failed rolling squared-error specification remains in v2, including all six failures. Gamma changes objective, link and selected hyperparameters, so a performance difference cannot be attributed solely to positivity.

## Temporal design

- Same trailing 1/5/22-session variance features for HAR and XGBoost.
- Direct HAR and XGBoost models for each horizon; persistence and EWMA multiplied by the horizon.
- Separate nested chronological selection by objective and horizon, with the original four-candidate budget.
- Monthly expanding fits; rolling Gamma uses the last 756 eligible rows and the expanding Gamma selection.
- Every training label must have matured at the fit date, including all five sessions of a multi-session target.
- Missing sessions invalidate full feature/target windows; no partial target sums or filling.
- Validation outcomes must mature before the outer year; all development targets end before 2018.

## Reproduce

```bash
python -m unittest discover -s tests -v
python -m volatility.horizons
```

Raw predictions, trial records and fit lineage are in artifacts/research_v3 (Git-ignored). Aggregate outputs are separate from earlier runs. If v1 predictions exist, the one-session reference forecasts are checked for agreement at rtol=1e-6, atol=1e-12. Existing v1/v2 file fingerprints are verified unchanged.

The next independent checks are additional instruments and eventually the reserved final period. This release does not claim tradable profits or a general ML advantage.

Source for the objective: [XGBoost parameter documentation](https://xgboost.readthedocs.io/en/stable/parameter.html).
'''
    (folder / 'research.md').write_text(note, encoding='utf-8')

