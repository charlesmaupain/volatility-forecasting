# Research protocol, draft 0.1

Written 26 October 2026 before market-data modelling.

## Question

Does XGBoost improve next-session realised-variance forecasts relative to HAR-RV when both use identical information? A statistical baseline winning is a valid research result.

## Forecast contract

- Initial target: next session's intraday realised variance, in squared decimal-return units, not annualised.
- Origin: after session t's variance measure becomes available. Document vendor publication delays; historical availability is not proof of point-in-time availability.
- Features: RV[t], mean of the five sessions ending at t, mean of the 22 sessions ending at t.
- Five-day extension: sum of next five session variances, not their mean or variance of the cumulative five-day return.
- Overnight variance is excluded initially. Economic evaluation must use compatible session-only exposure or extend the target to include overnight risk.
- Audit calendar gaps upstream. Do not silently forward-fill targets or treat missing sessions as consecutive.
- A training row is eligible only if its full target is available at fitting time. After-close fitting at t can use labels ending at t, but never later.

## Fair comparison

Start with persistence, fixed-decay EWMA, HAR in variance levels and XGBoost. Compare HAR and XGBoost with identical features before adding other information. Select EWMA decay in advance or inside training only. Use a documented positive forecast floor chosen without test information. Log-target experiments must handle retransformation explicitly: exponentiating expected log variance does not recover expected variance.

## Validation

After the data audit, freeze exact dates before examining model performance. Reserve the latest contiguous period for final evaluation. Part 1 and Part 2 use development periods only; Part 3 opens the final holdout once after choices are frozen.

Use expanding-window evaluation and inner chronological validation for tuning. Apply label-availability restrictions at every inner and outer boundary. Fit transformations only within training windows. For multiple instruments, use common calendar splits. No random cross-validation or test-driven early stopping.

Record fit dates, training cutoffs, bounded tuning budget, seeds, refitting frequency and runtime. Keep a log of all tried configurations and negative results.

## Evaluation

Primary loss: QLIKE = log(forecast) + observed / forecast. Observed variance can be zero; forecast variance must be positive. Use common targets and units. Secondary loss: variance MSE. Specify treatment of zeros before adding log-scale error.

Compare paired forecasts on identical dates and report sample size. Part 2 adds time-block bootstrap intervals or justified serial-correlation adjustments. Overlapping five-day targets and cross-instrument dependence invalidate naive independent-observation tests.

Predeclare regimes and ablations before final testing. Feature importance is descriptive, not causal. A HAR correction model must learn from historical out-of-sample HAR errors generated within training, not optimistic fitted residuals.

## Reproducibility

Record data URL, revision, retrieval timestamp, SHA-256, units, schema, filtering and counts. Save forecast origin, target end, model, training cutoff, prediction and outcome for every prediction. Keep raw files immutable and out of Git by default.

The initial benchmark includes empirical results and commands to reproduce the experiment.
