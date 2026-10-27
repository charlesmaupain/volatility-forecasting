import unittest
import numpy as np
import pandas as pd
from volatility.data import make_frame, session_dates
from volatility.experiment import training_at, loss, run
from volatility.temporal import qlike


class PipelineIntegrityTests(unittest.TestCase):
    def series(self):
        return pd.Series(np.linspace(0.0001, 0.0002, 100), index=pd.bdate_range('2010-01-01', periods=100))

    def test_midnight_labels_preserve_session_date_across_dst(self):
        result = session_dates(pd.Series(['2000-04-14 00:00:00+01:00', '2000-01-03 00:00:00+00:00']))
        self.assertEqual(result.dt.strftime('%Y-%m-%d').tolist(), ['2000-04-14', '2000-01-03'])

    def test_event_timestamps_are_not_silently_accepted_as_labels(self):
        with self.assertRaises(ValueError):
            session_dates(pd.Series(['2000-04-14 16:00:00+01:00']))

    def test_missing_session_is_not_compressed_or_imputed(self):
        series = self.series()
        series.iloc[40] = np.nan
        frame = make_frame(series)
        # Origin 39 has an unavailable target; origins 40..61 have incomplete features.
        self.assertTrue(all(date not in frame.index for date in series.index[39:62]))
        self.assertIn(series.index[62], frame.index)
        self.assertEqual(frame.loc[series.index[38], 'target_end'], series.index[39])

    def test_future_changes_do_not_change_eligible_training_rows(self):
        original = self.series()
        changed = original.copy()
        changed.iloc[51:] = 100.0
        cutoff = original.index[50]
        pd.testing.assert_frame_equal(training_at(make_frame(original), cutoff),
                                      training_at(make_frame(changed), cutoff))
        self.assertEqual(make_frame(original).loc[cutoff, 'daily'],
                         make_frame(changed).loc[cutoff, 'daily'])

    def test_label_matures_at_next_session_not_same_session(self):
        frame = make_frame(self.series())
        cutoff = frame.index[20]
        eligible = training_at(frame, cutoff)
        self.assertNotIn(cutoff, eligible.index)
        self.assertEqual(eligible.target_end.max(), cutoff)

    def test_normalised_qlike_preserves_paired_loss_differences(self):
        observed = np.array([0.1, 0.4, 0.01])
        first = np.array([0.12, 0.2, 0.02])
        second = np.array([0.2, 0.3, 0.03])
        expected = [qlike(o, a) - qlike(o, b) for o, a, b in zip(observed, first, second)]
        np.testing.assert_allclose(loss(observed, first) - loss(observed, second), expected)

    def test_holdout_boundary_fails_before_model_fitting(self):
        frame = make_frame(self.series())
        cfg = {'development_start': '2010-01-01', 'development_end': '2010-12-31',
               'holdout_start': '2010-03-01'}
        with self.assertRaisesRegex(ValueError, 'holdout'):
            run(frame, cfg, progress=False)


if __name__ == '__main__':
    unittest.main()
