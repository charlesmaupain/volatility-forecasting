# Positive forecasts and multiple horizons: v3 protocol

Recorded 27 October 2026 before running these specifications. This is exploratory follow-up to the negative rolling-XGBoost forecasts in v2. Preserve v1 and v2, including failed outcomes. No final-holdout evaluation.

## Questions

Does Gamma regression with a log link remove the negative-variance failure? Does its forecasting performance improve on HAR at one- and five-session horizons? Positivity alone is not evidence of accuracy.

## Objective and estimand

XGBoost reg:gamma produces a positive conditional-mean forecast through a log link. For fixed Gamma shape, its mean-dependent negative log likelihood is proportional to log(mu) + y/mu; this matches the QLIKE optimisation target up to observation-only terms and a positive constant. Treat this as a quasi-likelihood modelling choice, not a claim that realised variance is exactly Gamma distributed. It does not require exponentiating a fitted log-target mean or a lognormal retransformation assumption.

Compare persistence, fixed-decay EWMA, level HAR, squared-error XGBoost, Gamma XGBoost, and Gamma XGBoost with 756 eligible training rows. For one-session results, verify recomputed original forecasts against v1. Compare one-session rolling Gamma with the already-recorded failed rolling squared-error specification in v2. The rolling Gamma model retains its expanding Gamma model's selected hyperparameters, matching v2's window-sensitivity design. Gamma changes the fitting objective and link as well as the positivity constraint; do not attribute all differences to positivity alone.

Each objective and horizon independently selects from the same four prespecified depth/tree-count configurations and uses the same two preceding validation years. Refit monthly. Preserve seed, feature scaling, numerical floor and EWMA decay from v1. Record raw positivity failures and floor hits. Gamma predictions that are nonpositive or nonfinite stop the run rather than being silently repaired.

## Five-session target and timing

The target is the SUM of realised intraday variances over t+1 through t+5, not their mean or the variance of the cumulative return. Features remain the trailing 1/5/22-session variance summaries. Fit HAR directly to this horizon. Persistence and EWMA predict five times their one-session forecasts.

Use the complete XNYS session index. Missing feature or target sessions invalidate the example. No partial sums or imputation. Every training row must have target_end <= fitting origin, including inner tuning folds. Inner validation targets must mature before the outer year. Development outcomes must end before 2018. Adjacent five-session targets overlap; do not treat their errors as independent.

## Reporting

Save outputs separately under artifacts/research_v3 and reports/research_v3. Compare models on identical dates within each horizon; report sample counts separately across horizons. Primary metric is normalised QLIKE, with variance MSE secondary. Use stationary bootstrap expected block lengths 5/20/60, 5,000 replicates, seed 314159. Report pointwise intervals; simultaneous bands apply only to the specified within-horizon family, not all research choices or both horizons jointly.

Data-dependent follow-up and repeated development use mean these are not independent confirmatory findings. Bootstrap intervals are conditional on saved forecast paths, with no model reselection/refitting inside replicates. Note assumptions about approximate stationarity and dependence.

The 2018-2019 holdout remains reserved. Additional instruments, feature ablations and economic backtesting are not prerequisites for an honest v0.1 research preview.

Reference: https://xgboost.readthedocs.io/en/stable/parameter.html (reg:gamma, Gamma regression with log link).
