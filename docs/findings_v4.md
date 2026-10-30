# Interpretation of the neural forecasting experiment

## Main result

The Gamma GLM has the lowest average QLIKE among the v4 comparisons at both horizons: 0.220174 at one session and 0.180610 at five sessions. The PyTorch ensemble achieves 0.223741 and 0.184920, compared with HAR's 0.241438 and 0.203017. The nonlinear model does not outperform the positive linear control on average.

The GLM also has lower variance MSE than HAR at both horizons. The neural ensemble has higher variance MSE than HAR at both horizons. An improvement under the primary loss is not dominance under every error metric.

## Paired uncertainty

At expected bootstrap block length 20:

| Contrast | One-session difference, 95% pointwise interval | Five-session difference, 95% pointwise interval |
| --- | --- | --- |
| Gamma GLM minus HAR | -0.021264 [-0.032537, -0.011881] | -0.022407 [-0.038520, -0.007582] |
| Neural ensemble minus HAR | -0.017697 [-0.028153, -0.008207] | -0.018098 [-0.034766, -0.001406] |
| Neural ensemble minus GLM | +0.003567 [+0.000444, +0.007702] | +0.004309 [-0.002847, +0.014634] |

Negative differences favour the first model. The GLM-HAR pointwise intervals remain below zero at all three block lengths and both horizons. The neural-HAR five-session interval crosses zero at block length 60. The one-session neural-minus-GLM pointwise intervals stay positive; the five-session comparison remains unresolved.

The primary block-20 simultaneous bands for the three comparisons against HAR include zero at both horizons. Those bands cover a comparison family within each horizon, not the full history of experiments. Pointwise evidence must not be presented as familywise or independent final-test superiority. All intervals condition on the saved forecast paths and omit model refitting and broader research-selection uncertainty.

## Regime and seed diagnostics

The neural model's average gain is concentrated in lower-volatility periods. In the high-volatility group it is worse than HAR: one-session QLIKE 0.218850 versus 0.213479, and five-session QLIKE 0.208091 versus 0.190034. Regime thresholds use past observations only. These are descriptive subgroup findings without separate hypothesis tests.

All three seeds are retained. Their aggregate QLIKE values are close, but this limited seed set does not establish robustness to all initialisations, architectures or training settings. The arithmetic-mean ensemble is not better than every member; no seed is selected after inspecting performance.

## Complexity and cost

The GLM has four fitted coefficients including its intercept. Each neural member has 357 trainable parameters, with three members per ensemble. Across both horizons there are 288 GLM fits and 864 neural fits. Measured fitting time is approximately 2.7 seconds for the GLM and 571 seconds for all neural members on this CPU environment. These timings include model-specific preprocessing and training; they are not universal hardware benchmarks or an inference-latency comparison. Total study runtime, including evaluation, was approximately 588 seconds.

No nonpositive or nonfinite forecasts occurred. Original benchmark predictions are unchanged. The 2018-2019 final holdout remains unevaluated.

## Implications

The evidence favours investigating representation and positive conditional-mean modelling before increasing network size. The GLM control shows that a simpler specification can capture the improvement in this experiment. It does not isolate which representation or regularisation choice caused the gain.

The next study should test instrument robustness and feature ablations under a recorded design. Additional neural tuning would be a new exploratory experiment, not a revision that replaces this result. A single pre-COVID cash-index study cannot establish a general forecasting edge or a tradable strategy.
