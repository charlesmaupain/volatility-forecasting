# Frozen first-experiment choices

Recorded before the first model run on 26 October 2026.

Instrument: S&P 500 cash index (.SPX), not SPY ETF. Target: Oxford-Man rv5, daily intraday realised variance from five-minute returns. Archive revision and SHA-256 are pinned in configs/development.json. Overnight risk is not included.

Coverage: 2000-01-03 to 2020-02-21. Initial history starts in 2000. Development forecast origins span 2006-2017 with all outcomes ending before 2018. Reserve 2018-2019 for Part 3. Exclude partial 2020; no COVID claims.

Every development year selects among exactly four XGBoost configurations using the preceding two calendar years as inner validation folds. Each inner fold uses expanding training history and parameters fixed for that validation year. Refit HAR and selected XGBoost at the first eligible forecast origin each month; features update each session. Final holdout cannot be evaluated through the development command.

XGBoost and HAR receive identical 1/5/22-session variance features. Both fit squared-error objectives in variance levels scaled by 10,000 for numerical stability; XGBoost configurations are selected by QLIKE. No log retransformation. Fixed EWMA decay 0.94. Common prediction floor 1e-10, declared before outcomes. These choices are provisional research choices, not retrospectively optimised constants.

Use an explicit XNYS session calendar. Missing sessions remain missing, and any example with incomplete trailing windows or target is excluded for all models. EWMA resets at missing observations. Unexpected dates stop the pipeline for investigation.

QLIKE in reports uses the nonnegative normalisation observed/forecast - log(observed/forecast) - 1. Paired differences match the protocol's raw QLIKE. Do not report percentages of raw QLIKE, which can be negative. No claims of statistical significance or profitability in Part 1; dependent-data inference and economic testing remain later work.

Timing assumption: archive's session variance is available after that session for prediction of the next session. Actual historical vendor publication timestamps are unavailable. This is a retrospective forecasting experiment, not proof of executable real-time performance.
