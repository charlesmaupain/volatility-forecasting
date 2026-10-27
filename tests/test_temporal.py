import unittest
from volatility.temporal import available_training, make_examples, qlike


class TemporalIntegrityTests(unittest.TestCase):
    def test_hand_calculated_alignment(self):
        row = make_examples(list(range(1, 41)), 5)[0]
        self.assertEqual((row.origin, row.target_end), (21, 26))
        self.assertEqual(row.features, (22, 20, 11.5))
        self.assertEqual(row.target, 23 + 24 + 25 + 26 + 27)

    def test_future_changes_cannot_change_past_features(self):
        original = make_examples(list(range(1, 101)))
        changed = make_examples(list(range(1, 52)) + [99999] * 49)
        for left, right in zip(original, changed):
            if left.origin <= 50:
                self.assertEqual(left.features, right.features)

    def test_overlapping_future_labels_are_excluded(self):
        rows = available_training(make_examples(list(range(1, 101)), 5), 50)
        self.assertEqual(rows[-1].origin, 45)
        self.assertTrue(all(row.target_end <= 50 for row in rows))

    def test_future_changes_leave_training_data_unchanged(self):
        original = list(range(1, 101))
        changed = original[:51] + [99999] * 49
        self.assertEqual(available_training(make_examples(original, 5), 50),
                         available_training(make_examples(changed, 5), 50))

    def test_no_incomplete_features_or_partial_targets(self):
        self.assertEqual(make_examples([1] * 22), [])
        rows = make_examples([1] * 30, 5)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[-1].target_end, 29)

    def test_invalid_inputs(self):
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                make_examples([1] * 30 + [value])
        for horizon in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                make_examples([1] * 30, horizon)

    def test_qlike_domain_and_ranking(self):
        self.assertEqual(qlike(0, 1), 0)
        self.assertLess(qlike(2, 2), qlike(2, 4))
        for forecast in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                qlike(1, forecast)


if __name__ == '__main__':
    unittest.main()
