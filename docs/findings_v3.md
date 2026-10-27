# What v3 adds to the research

The Gamma/log-link specification addresses a mathematical constraint: a variance forecast must be positive. Its conditional-mean objective is aligned with QLIKE up to terms that do not affect the prediction optimum. Positive output does not guarantee accurate forecasting.

## Results

- One session: Gamma XGBoost QLIKE 0.237846; HAR 0.241438. The difference is -0.003592, with pointwise 95% interval [-0.023202, +0.022805] at expected block length 20. All tested block-length intervals contain zero.
- Five sessions: HAR QLIKE 0.203017; Gamma XGBoost 0.211856. Gamma-minus-HAR difference is +0.008838, with interval [-0.015470, +0.046278]. Again, all tested block-length intervals contain zero.
- HAR retains lower variance MSE than Gamma XGBoost at both horizons. Metric choice changes which small average advantages look attractive; primary QLIKE was recorded in advance.
- Expanding and rolling Gamma produced no nonpositive raw forecasts and no floor hits at either horizon. The six failed rolling squared-error forecasts from v2 remain preserved.
- The rolling Gamma version is worse on average than expanding Gamma at both horizons. Positivity repairs the invalid-output failure in these runs; it does not establish a rolling-window advantage.

## What is new about the five-session evaluation

The target sums t+1 through t+5 intraday variances. The label becomes available only at the fifth session. Training and tuning enforce that timing. Missing observations invalidate the whole target, not just a partial contribution. The 3,015 forecast dates differ from the 3,019 one-session dates because the last four origins cannot form complete development-period targets.

Adjacent outcomes overlap, so the uncertainty calculation uses dependent blocks. Those intervals still rely on approximate-stationarity assumptions and do not refit models inside bootstrap samples.

## Interpretation

A reproducible chronological benchmark finds strong statistical baselines competitive with nonlinear models. A negative-variance failure motivates a positive-output objective; the revised model avoids that failure, while its small average accuracy differences from HAR remain uncertain. The effect of horizon and objective is documented without opening the final holdout.

## Limitations

These experiments do not establish a general ML advantage, a profitable strategy, cross-market robustness, performance during COVID, or final-test superiority. Gamma uses a different objective/link and its own equally budgeted historical tuning; this is not an isolated causal test of positivity alone.

## Execution note

The first v3 execution completed model fits but failed the final reference-index assertion because in-memory dates used nanosecond resolution while CSV-parsed dates used microsecond resolution. The comparison was corrected to a common precision and the unchanged experiment rerun. No dates, forecasts, candidate choices or tuning budgets were changed in response. The original one-session forecasts agree within recorded numerical tolerances, and v1/v2 fingerprints are unchanged.
