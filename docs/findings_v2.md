# Interpretation of the second development experiment

## What the evidence supports

1. **Original HAR versus XGBoost:** mean QLIKE difference (XGBoost minus HAR) is +0.0222. Its pointwise 95% stationary-bootstrap interval is [+0.0113, +0.0358] with expected block length 20. Intervals also stay above zero at lengths 5 and 60. This supports a HAR advantage conditional on this historical forecast path and the bootstrap assumptions. It is not an independent final-test finding or a statement of familywise superiority.
2. **HAR correction:** mean QLIKE falls from 0.2414 to 0.2359, but the correction-minus-HAR interval is [-0.0140, +0.0024]. All three block-length intervals include zero. The hybrid is a research candidate, not an established improvement. It also has worse variance MSE than HAR.
3. **Rolling windows:** a 756-observation HAR window worsens average QLIKE to 0.2658. In the already-selected 2017 case study it worsens HAR from 0.3241 to 0.4780, so this experiment does not support the simple claim that a shorter window fixes slow adaptation.
4. **Session-aligned GARCH:** QLIKE is 0.2947, better than persistence and EWMA but worse than HAR. All 144 monthly fits converged. Some fitted persistence values are near one. This comparator uses session returns, not HAR's realised-variance features, and starts its contiguous return history after the last missing session.
5. **Rolling XGBoost fails a positivity requirement:** six negative raw variance predictions hit the predeclared floor. They produce mean QLIKE of 3895.1096. These outcomes are retained. The configuration is not suitable as a risk forecast in this form.

## Diagnosed failure, not a removed outlier

Reconstructing the affected fits confirmed negative raw predictions for origins 2015-09-18 and 2016-01-06, 08, 11, 12 and 13. Squared-error boosting does not enforce positive variance forecasts. The existing 1e-10 floor prevents invalid QLIKE arithmetic but cannot make a severely underestimated variance economically sensible.

A diagnostic counter initially missed those floor hits because XGBoost returned float32 values whose representation was slightly above the float64 floor. Counting with tight relative tolerance fixes reporting only; predictions and losses were not changed. A regression test covers this case.

The six extreme losses make the prespecified unstudentised simultaneous max-deviation bands extremely wide for every comparison. Those bands are disclosed as uninformative rather than removed or replaced after seeing the result. The separate pointwise intervals must not be presented as familywise significance.

## What the diagnostics add

The original XGBoost-HAR gap is larger on the 120 dates outside the original monthly training feature ranges (0.4849 versus 0.2908 QLIKE) than on the 2,899 in-range dates (0.2545 versus 0.2394). This is consistent with an extrapolation concern, but it is not causal proof. Rolling-model training ranges differ and are not measured by that original-model diagnostic.

The correction improves the low-volatility group (0.2245 versus HAR 0.2477), while worsening the high-volatility group (0.2315 versus 0.2135). Regimes use only lagged historical thresholds and information available at each forecast origin. The overall small advantage masks differing conditional behaviour.

## Next bounded experiments

The follow-up questions concern positive-output modelling, a five-day target and additional instruments. Subsequent experiments retain the failed rolling specification as a reference. Any log-variance specification requires conditional-mean retransformation. The final 2018-2019 period remains reserved until model choices are frozen.

These extensions were chosen after seeing v1 development results. 
