# Research roadmap

## Implemented foundation

The initial benchmark establishes data provenance, session-calendar alignment, matured-label checks, nested chronological XGBoost selection and statistical baselines. Follow-up experiments add dependent-data uncertainty, rolling windows, an out-of-sample HAR correction, positive-output objectives and two forecast horizons.

## Neural conditional-mean study

Experiment v4 tests a compact PyTorch network trained directly on QLIKE. A Gamma GLM shares its log-transformed inputs and positive link. Three fixed seeds and their ensemble measure optimisation sensitivity. Monthly fits, exact reference alignment, bootstrap intervals and compute records make this a reproducible empirical comparison. Architecture choices are fixed before the experiment; additional tuning would require a separately recorded study.

## Cross-instrument robustness and feature ablations

The next study will select a small, explicitly declared set of instruments from the same archive, audit each exchange calendar and session convention, and evaluate models independently on matched development dates. Cross-market features will require actual availability alignment, not same-date joins across time zones. No pooled model will use outcomes that have not matured at the receiving instrument's forecast time.

Feature ablations will compare daily-only, daily/weekly/monthly and separately justified additional inputs with matched tuning budgets. Paired resampling across instruments must preserve shared-date dependence; a large number of instrument-days is not an equally large number of independent observations. Model complexity and fit cost will be reported alongside accuracy.

A sequence model is a possible later extension if individual lagged observations add evidence beyond aggregated HAR features. It is not yet implemented. A larger architecture will need a suitable sample size and an explicit advantage to test.

## Frozen final evaluation

After development choices are complete, record the final data transformations, model specifications, selection rules, loss functions and primary comparisons. Open the reserved 2018-2019 period once under that specification. Preserve all outcomes and distinguish confirmatory comparisons from subsequent exploratory analysis.

## Economic evaluation

A separate risk-control study will connect forecast variance to position sizing, with explicit information cutoffs, tradable prices, overnight exposure, leverage limits, transaction costs and a simple comparison policy. The present cash-index intraday variance series is not by itself an executable strategy. Data acquisition and timing checks are prerequisites.

## Engineering

Keep original experiments reproducible, optional model dependencies separate, and forecast lineage inspectable. Extend CI to cover each new leakage boundary. Any dashboard will present completed evidence and limitations rather than substitute for a reproducible research report.
