# Part 2 research protocol: spx-rv5-research-v2

Written before running the new specifications. This is exploratory follow-up to the already inspected 2006-2017 development results, not a new independent confirmation. Preserve v1 files and never access final holdout outcomes for model comparison.

## Uncertainty

Use paired date-aligned QLIKE differences, model minus expanding HAR. Stationary bootstrap resamples entire vectors of model loss differences together, preserving pairing and short-range dependence. Use 5,000 replicates, expected block lengths 5/20/60 sessions (20 primary), seed 271828. Report pointwise 95% percentile intervals and joint max-deviation bands for the finite reported comparison family. No bootstrap fraction is called a p-value. Intervals are conditional on the fitted forecast paths: no refitting or reselection inside bootstrap. Approximate stationarity/dependence assumptions and development-selection limitations must be explicit.

## Diagnostics

Report performance by year; 2008 and 2017 were selected after seeing v1 and remain descriptive. Assign low/medium/high volatility regimes using current weekly RV and expanding 25th/75th percentile thresholds of prior weekly RV only. At each original monthly fit, classify whether any of the three features lies outside the training range. Neither diagnostic proves causality.

## Rolling sensitivity

Compare the last 756 eligible training observations against expanding history for HAR and XGBoost. Keep first-experiment hyperparameters selected from past data fixed for the rolling XGBoost comparison, isolating the training-window effect; this is not optimal rolling-model tuning. The 756 rows approximate three trading years and can span longer calendar time when windows are excluded.

## HAR correction

Generate monthly expanding HAR forecasts starting in 2003 after at least 500 matured training examples. Store these historical out-of-sample predictions and their target-end dates. At each outer fit, train a fixed small XGBoost model on matured out-of-sample errors (actual variance minus HAR forecast), using the original three features only. Add the predicted correction to the current expanding HAR forecast; enforce the same positive floor. Require at least 252 matured residuals. No in-sample HAR residuals, retrospective corrected historical forecasts or final-test tuning.

## GARCH session-risk comparator

Fit zero-mean GARCH(1,1) with Student-t innovations to open-to-close log returns, in percent units, and forecast next-session variance divided by 10,000. Verify the archive return field against log(close/open). This matches the session exposure more closely than close-to-close returns but is still a conditional return-variance model evaluated against a noisy realised-variance proxy, under a zero conditional mean assumption. It uses a different information set, so it is not evidence about nonlinear versus linear methods on matched features.

Do not compress missing return sessions: fit only the trailing contiguous observed return segment, at least 252 observations. Refit monthly and recursively update conditional variance with each newly observed session return. Log fit convergence, parameter persistence, training start/end and failures. Stop on fit failure rather than silently dropping difficult dates.

## Outputs and integrity

Store new outputs under artifacts/research_v2 and reports/research_v2. Fingerprint the original predictions and refuse to change them. Every added model must forecast every original evaluation date; no model-specific omission. Record candidate and feature configuration, versions, source hash and original forecast hash. Keep raw data and row-level outcomes outside Git. Multi-instrument and five-day extensions remain subsequent Part 2 work.
