import unittest
import numpy as np
import pandas as pd
from volatility.data import make_frame
from volatility.experiment import run


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.bdate_range('2000-01-03', '2007-12-31')
        rng = np.random.default_rng(53)
        original = pd.Series(np.exp(rng.normal(-9, .4, len(dates))), index=dates)
        changed = original.copy()
        changed.loc['2007-01-01':] *= 5
        cfg = {'development_start': '2006-01-01', 'development_end': '2007-12-31',
               'holdout_start': '2008-01-01', 'forecast_floor': 1e-10,
               'scale': 10000, 'seed': 1729, 'learning_rate': .03,
               'xgb_candidates': [{'max_depth': 2, 'n_estimators': 5}]}
        cls.first, cls.trials, _ = run(make_frame(original), cfg, progress=False)
        cls.second, _, _ = run(make_frame(changed), cfg, progress=False)

    def test_future_data_cannot_change_prior_model_forecasts(self):
        first = self.first[self.first.target_end < '2007-01-01'].reset_index(drop=True)
        second = self.second[self.second.target_end < '2007-01-01'].reset_index(drop=True)
        pd.testing.assert_frame_equal(first, second)

    def test_prediction_and_tuning_lineage(self):
        p = self.first
        self.assertTrue((p.training_target_max <= p.fit_origin).all())
        self.assertTrue((p.fit_origin <= p.origin).all())
        self.assertTrue((p.origin < p.target_end).all())
        self.assertTrue((p.forecast > 0).all())
        self.assertEqual(p.groupby('model').size().nunique(), 1)
        self.assertEqual(p.groupby('origin').model.nunique().min(), 4)
        for trial in self.trials:
            self.assertLess(trial['validation_target_max'], str(trial['outer_year']) + '-01-01')
            self.assertLessEqual(trial['latest_training_target'], trial['fit_origin'])


if __name__ == '__main__':
    unittest.main()
