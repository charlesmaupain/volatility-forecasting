"""Static research figures and a generated, numerical v4 report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .data import ROOT


def render(metrics, intervals, annual, metadata):
    destination = ROOT/'reports/research_v4'
    figures = destination/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    names = ['HAR','XGBoost Gamma','Gamma GLM','MLP ensemble']
    colors = ['#64748b','#d97706','#0f766e','#4338ca']
    fig, axes = plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for ax,h in zip(axes,[1,5]):
        subset = metrics[metrics.horizon==h].set_index('model')
        values = subset.loc[names,'qlike']
        ax.barh(names,values,color=colors,height=.6)
        ax.invert_yaxis()
        ax.set_xlim(0,max(values)*1.23)
        for i,value in enumerate(values):
            ax.text(value+.003,i,f'{value:.4f}',va='center')
        ax.set_title(f'{h}-session horizon')
        ax.set_xlabel('Mean QLIKE (lower is better)')
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('Can a small neural network improve volatility forecasts?',fontweight='bold')
    for ext in ['png','svg']:
        fig.savefig(figures/f'neural_comparison.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,h in zip(axes,[1,5]):
        subset=intervals[(intervals.horizon==h)&(intervals.reference=='HAR')&(intervals.block_length==20)].set_index('model').loc[names[1:]]
        for i,(name,row) in enumerate(subset.iterrows()):
            ax.plot([row.pointwise_low,row.pointwise_high],[i,i],color=colors[i+1],linewidth=3)
            ax.scatter(row.mean_difference,i,color=colors[i+1],s=45,zorder=3)
        ax.axvline(0,color='#64748b',linestyle='--')
        ax.set_yticks(range(3),names[1:])
        ax.invert_yaxis()
        ax.set_xlabel('QLIKE difference from HAR (negative favours model)')
        ax.set_title(f'{h}-session horizon')
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('Pointwise 95% intervals | stationary bootstrap, block length 20',fontweight='bold')
    for ext in ['png','svg']:
        fig.savefig(figures/f'neural_uncertainty.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,h in zip(axes,[1,5]):
        pivot=annual[annual.horizon==h].pivot(index='year',columns='model',values='qlike')
        seedcols=[c for c in pivot if c.startswith('MLP seed')]
        delta=pivot[seedcols].subtract(pivot.HAR,axis=0)
        ax.fill_between(pivot.index,delta.min(axis=1),delta.max(axis=1),alpha=.2,color=colors[-1],label='Range across 3 seeds')
        ax.plot(pivot.index,pivot['MLP ensemble']-pivot.HAR,color=colors[-1],label='Ensemble minus HAR')
        ax.axhline(0,color='#64748b',linestyle='--')
        ax.set_title(f'{h}-session horizon')
        ax.set_ylabel('Annual mean QLIKE difference')
        ax.legend(fontsize=8)
    fig.suptitle('Initialisation sensitivity over time | range is not a confidence interval',fontweight='bold')
    for ext in ['png','svg']:
        fig.savefig(figures/f'seed_sensitivity.{ext}',dpi=180)
    plt.close(fig)
    lines=['# Neural volatility forecasting: development experiment v4','',
        'A fixed small PyTorch network trained on QLIKE is compared with a positive Gamma GLM, HAR and the existing Gamma XGBoost. No architecture or seed was selected using these results.','',
        '![Model comparison](figures/neural_comparison.png)','',
        '## Results','', '| Horizon | Model | QLIKE | Variance MSE | Forecasts |','| --- | --- | ---: | ---: | ---: |']
    for row in metrics.sort_values(['horizon','qlike']).itertuples():
        lines.append(f'| {row.horizon} | {row.model} | {row.qlike:.6f} | {row.mse:.6g} | {row.observations} |')
    lines+=['','## Paired uncertainty','', 'Negative differences favour the model. These are pointwise intervals, conditional on saved forecast paths. Simultaneous family bands and block-length sensitivity are in uncertainty.csv. The GLM contrast is a separate exploratory comparison.','',
            '| Horizon | Model | Reference | Difference | 95% interval, block 20 |','| --- | --- | --- | ---: | --- |']
    for row in intervals[intervals.block_length==20].itertuples():
        lines.append(f'| {row.horizon} | {row.model} | {row.reference} | {row.mean_difference:+.6f} | [{row.pointwise_low:+.6f}, {row.pointwise_high:+.6f}] |')
    lines+=['','![Paired uncertainty](figures/neural_uncertainty.png)','',
            '## Seed sensitivity','', 'All three fixed seeds are reported. The ensemble averages variance predictions, not log predictions. Seed dispersion is optimisation sensitivity, not calibrated forecast uncertainty.', '',
            '![Seed sensitivity](figures/seed_sensitivity.png)','',
            '## Design and limits','',
            'The network has two 16-unit tanh layers and a linear skip connection (357 parameters per seed). Training uses 200 full-batch AdamW epochs per monthly refit. Feature standardisation and target scaling use matured training rows only. Outputs are positive conditional means trained with QLIKE, not exponentiated squared-error predictions of log variance.', '',
            'The GLM shares the neural feature representation and positive link, but has different regularisation and optimisation. HAR and Gamma XGBoost are unchanged v3 references. Equal underlying inputs do not imply identical training or tuning budgets: the neural/GLM specifications are fixed; XGBoost was selected chronologically.', '',
            'The study remains one instrument, 2006-2017 development data reused across experiments. No final holdout, trading profitability, COVID robustness or general neural superiority is established. Bootstrap intervals do not include refitting or all research-selection uncertainty. CPU determinism is environment-specific.', '',
            f"Total experiment runtime: {metadata['elapsed_seconds']:.1f} seconds. Previous benchmark outputs unchanged: {metadata['previous_outputs_unchanged']}. Final holdout evaluated: {metadata['holdout_evaluated']}.", '',
            'See [protocol](../../docs/research_v4_protocol.md), [compute costs](compute.csv), [annual results](annual_metrics.csv), [regime results](regime_metrics.csv), and [uncertainty](uncertainty.csv).']
    (destination/'research.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
