import unittest
import numpy as np
import pandas as pd
from volatility.horizons import horizon_frame, model, forecast, run_horizon
from volatility.data import make_frame
from volatility.experiment import training_at


class HorizonIntegrityTests(unittest.TestCase):
    def series(self, n=100):
        return pd.Series(np.arange(1, n+1, dtype=float), index=pd.bdate_range('2000-01-03', periods=n))

    def test_one_session_frame_matches_original(self):
        series = self.series()
        pd.testing.assert_frame_equal(horizon_frame(series, 1), make_frame(series))

    def test_five_session_sum_and_exact_target_end(self):
        series = self.series(40)
        frame = horizon_frame(series, 5)
        row = frame.iloc[0]
        self.assertEqual(row.target, 23+24+25+26+27)
        self.assertEqual(row.target_end, series.index[26])
        self.assertEqual(row.daily, 22)
        self.assertEqual(row.weekly, 20)
        self.assertEqual(row.monthly, 11.5)
        self.assertEqual(len(frame), 14)
        self.assertEqual(frame.iloc[-1].target_end, series.index[-1])

    def test_no_partial_future_sum_when_a_session_is_missing(self):
        series = self.series()
        series.iloc[40] = np.nan
        frame = horizon_frame(series, 5)
        self.assertTrue(all(date not in frame.index for date in series.index[35:62]))
        self.assertIn(series.index[34], frame.index)
        self.assertIn(series.index[62], frame.index)

    def test_five_session_training_purges_unmatured_labels(self):
        series = self.series()
        frame = horizon_frame(series, 5)
        origin = series.index[50]
        training = training_at(frame, origin)
        self.assertEqual(training.index.max(), series.index[45])
        self.assertEqual(training.target_end.max(), origin)

    def test_gamma_predictions_remain_positive_on_extreme_features(self):
        frame = horizon_frame(self.series(), 1)
        cfg = {'scale': 10000, 'forecast_floor': 1e-10, 'learning_rate': .03, 'seed': 7}
        fitted = model({'max_depth': 2, 'n_estimators': 10}, 'reg:gamma', cfg).fit(
            frame[['daily', 'weekly', 'monthly']].to_numpy()*cfg['scale'],frame.target.to_numpy()*cfg['scale'])
        extreme = frame.iloc[:2].copy()
        extreme.loc[:, ['daily','weekly','monthly']] = [[1e-12]*3, [1e10]*3]
        values, failures = forecast(fitted, extreme, cfg, positive=True)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values > 0).all())
        self.assertEqual(failures, 0)

    def test_gamma_invalid_output_is_not_silently_clipped(self):
        class Broken:
            def predict(self, x):
                return np.zeros(len(x))
        with self.assertRaisesRegex(ValueError, 'positive-output'):
            forecast(Broken(), horizon_frame(self.series(), 1), {'scale':10000, 'forecast_floor':1e-10}, positive=True)

    def test_five_session_full_pipeline_future_invariance_and_tuning_purge(self):
        dates = pd.bdate_range('2000-01-03', '2006-03-31')
        series = pd.Series(np.exp(np.random.default_rng(8).normal(-9,.3,len(dates))),index=dates)
        cfg = {'development_start':'2006-01-01','development_end':'2006-03-31','holdout_start':'2008-01-01',
               'scale':10000,'forecast_floor':1e-10,'seed':7,'learning_rate':.03,
               'xgb_candidates':[{'max_depth':2,'n_estimators':5}]}
        protocol = {'rolling_observations':756}
        first, trials, _ = run_horizon(horizon_frame(series,5),5,cfg,protocol,progress=False)
        changed = series.copy()
        changed.loc['2006-02-01':] *= 5
        second, _, _ = run_horizon(horizon_frame(changed,5),5,cfg,protocol,progress=False)
        pd.testing.assert_frame_equal(first[first.target_end<'2006-02-01'].reset_index(drop=True),
                                      second[second.target_end<'2006-02-01'].reset_index(drop=True))
        self.assertTrue((first.training_target_max <= first.fit_origin).all())
        self.assertTrue((first.fit_origin <= first.origin).all())
        self.assertEqual(first.groupby('origin').model.nunique().min(),6)
        for trial in trials:
            self.assertLessEqual(trial['training_target_max'], trial['fit_origin'])
            self.assertLess(trial['validation_target_max'], str(trial['outer_year'])+'-01-01')


if __name__ == '__main__':
    unittest.main()
