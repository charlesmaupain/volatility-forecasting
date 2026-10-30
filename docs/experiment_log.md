# Experiment log

## spx-rv5-development-v1 — 26 October 2026

First empirical model run. Configuration was written before fitting. Development years 2006-2017; 2018-2019 reserved. Four XGBoost candidates, selected separately for each outer year on the previous two inner validation years. All trial losses are saved in artifacts/development/tuning.json.

Data audit caught a civil-date versus UTC conversion bug before modelling. Fixed by parsing the stated date of validated midnight labels. Added a DST regression test. Fourteen actual missing sessions remained; no unexpected sessions remained. No statistical results were inspected before this correction.

Result: HAR QLIKE 0.241438; XGBoost 0.263633; EWMA 0.304207; persistence 0.304929. Each model has 3,019 forecasts. No model changes were made in response to these results. Final holdout remains unevaluated.

Later documentation, report-layout edits and integrity tests do not constitute alternative model experiments. Changes to data, features, fitting windows, objectives or hyperparameter budgets are recorded as separate experiments.

## spx-rv5-research-v2 — 26 October 2026

Protocol: docs/research_v2_protocol.md, written before the new fits. Exploratory follow-up on the same development period. Original forecasts and fit records retained and fingerprinted.

Added 756-row rolling HAR and XGBoost, fixed-configuration XGBoost correction trained on historical out-of-sample HAR errors, and zero-mean Student-t GARCH on session returns. All eight models forecast the same 3,019 dates. All 144 GARCH fits converged.

Mean QLIKE: HAR correction 0.235861; original HAR 0.241438; original XGBoost 0.263633; rolling HAR 0.265771; session GARCH 0.294715; EWMA 0.304207; persistence 0.304929; rolling XGBoost 3895.109636.

Stationary bootstrap: 5,000 replicates at expected block lengths 5, 20 and 60. The hybrid gain is not resolved by pointwise intervals; the original XGBoost-HAR gap stays positive. The unstudentised family bands are dominated by the failed rolling specification and remain uninformative.

Investigated six negative raw rolling-XGBoost forecasts and retained their floored outcomes. Corrected the floor-hit diagnostic for float32 representation; no predictions, model choices or loss values changed. Improved figure layout after visual review. No retuning, dropped failure dates or final-holdout evaluation.

## spx-positive-multihorizon-v3 — 27 October 2026

Prespecified positive-output and direct-horizon experiment (docs/research_v3_protocol.md). Same four-candidate search budget per objective and horizon, with monthly fits and matured labels throughout nested validation. Development only; final holdout remains unevaluated.

One-session QLIKE: Gamma XGBoost 0.237846; HAR 0.241438; rolling Gamma 0.254700; squared-error XGBoost 0.263633; EWMA 0.304207; persistence 0.304929. Each model has 3,019 forecasts.

Five-session QLIKE: HAR 0.203017; Gamma XGBoost 0.211856; squared-error XGBoost 0.214771; EWMA 0.237917; rolling Gamma 0.240335; persistence 0.343647. Each model has 3,015 forecasts. Gamma had no nonpositive outputs or floor hits in either window/horizon configuration. Its paired difference from HAR remains unresolved by the bootstrap intervals at both horizons.

Initial execution failed a final verification assertion on datetime64[ns] versus datetime64[us], after model fitting but before report publication. Normalised date precision in the assertion and reran unchanged settings. Reference one-session forecasts reproduced within tolerance; v1/v2 forecast and fit fingerprints remain unchanged.

Release verification: installed the lock file in a fresh Windows/Python 3.13 environment, ran all 32 tests, downloaded the pinned archive afresh and regenerated v1. Its 12,076 model/date forecast rows are byte-identical to the original CSV. Full v2/v3 empirical runs were not repeated in the clean environment. GitHub-hosted CI has not yet run.

## spx-neural-qlike-v4 — 29 October 2026

Protocol and settings recorded before empirical fitting. Added a fixed 357-parameter PyTorch network, three declared seeds, an arithmetic-mean ensemble and a Gamma GLM control with matched log-transformed features. Monthly expanding fits use matured labels and training-only standardisation at both horizons. No hyperparameter search, early stopping or best-seed selection.

Completed 1,152 monthly fits, producing 42,238 combined model/origin records including unchanged HAR and Gamma XGBoost references. All forecasts are finite and positive. One-session QLIKE: GLM 0.220174, neural ensemble 0.223741, HAR 0.241438. Five-session QLIKE: GLM 0.180610, neural ensemble 0.184920, HAR 0.203017. The neural model has worse MSE and high-volatility-group QLIKE than HAR at both horizons. Findings and all seeds are retained.

Pointwise block-20 intervals favour both new models over HAR, but simultaneous family bands include zero. The five-session neural-HAR pointwise interval includes zero at block length 60. GLM remains the stronger average performer. Clarified that family bands cover the comparisons within each horizon; the calculation and experiment settings were unchanged.

Validation: all 40 tests passed; eight optional neural tests explicitly skip in the original environment without PyTorch. A first synthetic repeatability test incorrectly required six AdamW steps on random data to reduce loss. Separated that assumption from reproducibility and added a learning test with a known positive conditional mean. This happened before market-data fitting and did not change the recorded empirical settings. Package consistency check passed. The full neural experiment was run in the existing local environment, not a newly rebuilt environment; hosted neural CI remains unverified until it runs.

All 864 neural fits reduced training objective from initialisation, but this is not a claim of optimisation convergence. Previous forecast hashes are unchanged. The final 2018-2019 holdout remains unevaluated. No subsequent model retuning was performed.
