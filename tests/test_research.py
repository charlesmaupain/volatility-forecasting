import unittest
import numpy as np
import pandas as pd
from volatility.inference import paired_losses, bootstrap_intervals, causal_regimes, floor_mask
from volatility.research_models import har_oof, matured_residuals, trailing_returns, variance_step, garch_block
from volatility.data import make_frame


class ResearchIntegrityTests(unittest.TestCase):
    def test_floor_counter_handles_float32_rounding_without_counting_nearby_values(self):
        rounded = float(np.float32(1e-10))
        self.assertGreater(rounded, 1e-10)
        np.testing.assert_array_equal(floor_mask([rounded, 1e-10, 1.01e-10], 1e-10), [True, True, False])
    def test_paired_alignment_rejects_missing_and_mismatched_outcomes(self):
        rows = pd.DataFrame({'origin': pd.to_datetime(['2000-01-03'] * 2 + ['2000-01-04'] * 2),
                             'target_end': pd.to_datetime(['2000-01-04'] * 2 + ['2000-01-05'] * 2),
                             'model': ['HAR', 'XGBoost'] * 2, 'observed': [1., 1., 2., 2.],
                             'forecast': [1., 1.2, 2., 2.2]})
        first, _ = paired_losses(rows)
        second, _ = paired_losses(rows.sample(frac=1, random_state=7))
        pd.testing.assert_frame_equal(first, second)
        with self.assertRaises(ValueError):
            paired_losses(rows.iloc[:-1])
        broken = rows.copy()
        broken.loc[1, 'observed'] = 9
        with self.assertRaises(ValueError):
            paired_losses(broken)

    def test_bootstrap_keeps_pairing_and_is_reproducible(self):
        differences = pd.DataFrame({'a': np.arange(60) / 100, 'b': np.arange(60) / 100})
        first = bootstrap_intervals(differences, 5, 200, seed=7)
        second = bootstrap_intervals(differences, 5, 200, seed=7)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.iloc[0].pointwise_low, first.iloc[1].pointwise_low)
        self.assertEqual(first.iloc[0].pointwise_high, first.iloc[1].pointwise_high)
        self.assertTrue((first.family_low <= first.mean_difference).all())
        self.assertTrue((first.family_high >= first.mean_difference).all())

    def test_constant_difference_has_degenerate_interval(self):
        result = bootstrap_intervals(pd.DataFrame({'a': [2.] * 80}), 5, 100)
        self.assertEqual(result.iloc[0].pointwise_low, 2.)
        self.assertEqual(result.iloc[0].pointwise_high, 2.)

    def test_regimes_cannot_see_future_values(self):
        dates = pd.bdate_range('2000-01-03', periods=500)
        first = pd.Series(np.linspace(.0001, .0002, 500), index=dates)
        changed = first.copy()
        changed.iloc[401:] = 1000
        origins = dates[300:401]
        pd.testing.assert_frame_equal(causal_regimes(first, origins), causal_regimes(changed, origins))

    def test_garch_gap_is_not_compressed(self):
        dates = pd.bdate_range('2000-01-03', periods=20)
        returns = pd.Series(.01, index=dates)
        returns.iloc[10] = np.nan
        result = trailing_returns(returns, dates[-1], 5)
        self.assertEqual(result.index.min(), dates[11])
        self.assertEqual(len(result), 9)
        with self.assertRaises(ValueError):
            trailing_returns(returns, dates[-1], 10)

    def test_garch_recursion_uses_current_return_once(self):
        self.assertAlmostEqual(variance_step(.1, .2, .7, 2., 3.), 3.)
        self.assertAlmostEqual(variance_step(.1, .2, .7, 1., 3.), 2.4)

    def test_har_residual_lineage_and_future_invariance(self):
        dates = pd.bdate_range('2000-01-03', periods=900)
        series = pd.Series(np.exp(np.random.default_rng(5).normal(-9, .4, 900)), index=dates)
        cfg = {'scale': 10000, 'forecast_floor': 1e-10}
        first = har_oof(make_frame(series), cfg, '2002-01-01')
        changed = series.copy()
        cutoff = dates[800]
        changed.iloc[801:] *= 5
        second = har_oof(make_frame(changed), cfg, '2002-01-01')
        eligible = matured_residuals(first, cutoff)
        pd.testing.assert_frame_equal(eligible, matured_residuals(second, cutoff))
        self.assertTrue((eligible.target_end <= cutoff).all())
        self.assertTrue((eligible.base_fit_origin <= eligible.index).all())
        self.assertNotIn(cutoff, eligible.index)

    def test_garch_later_returns_do_not_change_earlier_predictions(self):
        dates = pd.bdate_range('2000-01-03', periods=500)
        returns = pd.Series(np.random.default_rng(8).standard_t(8, 500) * .01, index=dates)
        changed = returns.copy()
        changed.iloc[405:] *= 5
        original, log = garch_block(returns, dates[400:410])
        alternative, _ = garch_block(changed, dates[400:410])
        np.testing.assert_allclose(original[:5], alternative[:5], rtol=1e-10)
        self.assertEqual(log['convergence_flag'], 0)
        self.assertTrue((original > 0).all())


if __name__ == '__main__':
    unittest.main()

