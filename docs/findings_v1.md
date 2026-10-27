# First-run interpretation

The first development experiment gives HAR the lowest average QLIKE (0.2414), followed by XGBoost (0.2636), EWMA (0.3042) and persistence (0.3049), on 3,019 common forecast origins. HAR also has the lowest variance MSE; EWMA beats XGBoost on MSE, illustrating why the primary loss was fixed beforehand.

HAR beats XGBoost in 11 of 12 development years. In 2015, XGBoost narrowly wins their paired comparison (0.3273 vs 0.3339). In 2017, EWMA is best of the four (0.2204), and HAR (0.3241) and XGBoost (0.3655) are worse than persistence (0.2332). These patterns are descriptive and were observed after the run; they are not predeclared hypothesis-test results.

## Hypotheses for Part 2, not conclusions

- Expanding fits may adapt slowly when volatility levels change. Test a prespecified rolling-window alternative using development data only.
- Trees may struggle to extrapolate beyond historical training ranges. Diagnose whether crisis errors coincide with out-of-range features before attributing the result to this mechanism.
- Squared-error fitting and QLIKE evaluation target different aspects of errors. Test a justified alternative objective without changing the original benchmark.
- Daily, weekly and monthly averages may already capture much of the useful persistence. Add feature groups separately and compare equal information sets.

No statistical significance is asserted. Part 2 needs dependent-data confidence intervals, careful multiple-experiment interpretation, and robustness across instruments. A negative ML result is preserved as evidence rather than hidden by retrospective retuning.
