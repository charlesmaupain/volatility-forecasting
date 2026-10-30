# Neural conditional-mean forecasts: experiment v4

Recorded 29 October 2026 before fitting the new models. This is an exploratory follow-up to v1-v3 on the same development period, not independent confirmation.

## Question and controls

Does a compact nonlinear model trained directly on QLIKE improve variance forecasts relative to HAR, Gamma XGBoost and a Gamma generalised linear model (GLM)? The GLM controls for a positive log link and logarithmic feature representation. The neural model uses the same underlying daily, weekly and monthly realised-variance information, transformed with log and standardised using each training window only. This is a comparison of complete specifications, not an isolated causal test of architecture: regularisation and optimisation differ.

## Frozen specification

- Same S&P 500 data, one- and five-session sum targets, 2006-2017 origins and complete-calendar treatment as v3. The 2018-2019 holdout remains unopened.
- Monthly expanding refits; training origins precede fitting and every training target has matured by the fit origin.
- Gamma GLM: log link, alpha 0.001, L-BFGS, max_iter 2000, tol 1e-8. Convergence warnings fail the experiment.
- PyTorch network: three inputs, two hidden layers of width 16 with tanh, linear output giving log conditional mean. A separate linear skip connection permits a log-linear component. Exponentiation produces the variance mean; the target is not log-transformed for a squared-error fit.
- Loss: mean(z + y_scaled * exp(-z)), where y_scaled is target divided by the training-target mean. This differs from QLIKE only by observation-only terms. Scaling is learned from training only.
- Full-batch AdamW, learning rate 0.01, weight decay 0.001, 200 epochs, gradient norm clipped to 5. No early stopping, hyperparameter search or development-result-driven epoch selection.
- Fixed seeds 17, 43 and 101; report all three and their arithmetic-mean forecast ensemble. No best-seed selection. Seed spread is optimisation sensitivity, not a predictive interval.
- CPU float64, one thread, deterministic algorithms. Reproducibility is environment-specific; cross-platform bitwise identity is not claimed.
- No output floor or clipping; nonfinite/nonpositive predictions fail explicitly. Every refit records training dates, scalers, seed, training loss and runtime.

## Evaluation

Compare with unchanged v3 HAR and Gamma XGBoost predictions on exactly matching origins, target ends and observations. Main comparisons are the GLM and fixed three-seed ensemble versus HAR; individual seeds are diagnostics. Report QLIKE, variance MSE, yearly and causal volatility-regime losses, training time and parameter count. Use paired stationary bootstrap with 5,000 repetitions, expected block lengths 5/20/60, primary length 20. Include pointwise and simultaneous family bands for the three comparisons against HAR within each horizon, conditional on these saved forecast paths. These bands do not jointly cover both horizons or prior experiments. Ensemble-minus-GLM differences are a separate exploratory paired contrast.

Overlapping five-session targets require dependent resampling. Intervals do not include model refitting or account for the full history of research selection. Small sample size, a single instrument, imperfect historical publication timing and the pre-COVID archive remain limitations. No claim that a neural model is better will be made solely because it has more parameters.

## Reproduction

Run the v3 experiment first, then install the optional neural requirements and run `python -m volatility.neural_experiment`. Detailed predictions and fit records stay in Git-ignored artifacts; aggregate reports and figures are public.

References: [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [GammaRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html).
