import unittest

import numpy as np
import pandas as pd

from gbdt import compute_stats


class PairStatisticsTests(unittest.TestCase):
    def test_pair_statistics_use_pair_exposures_and_global_smoothing(self):
        frame = pd.DataFrame({'user_id': ['u', 'u', 'u', 'v'],
                              'item_id': ['a', 'a', 'b', 'a'],
                              'label': [1, 0, 0, 0]})
        items, users, pairs = compute_stats(frame, 0.25)
        self.assertEqual(len(items), 2)
        self.assertEqual(len(users), 2)
        self.assertEqual(len(pairs), 3)
        pair = pairs.set_index(['user_id', 'item_id']).loc[('u', 'a')]
        self.assertAlmostEqual(pair['user_item_ctr'], (1 + 5 * 0.25) / (2 + 5))
        self.assertAlmostEqual(pair['ui_log_exposures'], np.log1p(2))
        self.assertEqual(pairs['user_item_ctr'].dtype, np.float32)
        self.assertEqual(pairs['ui_log_exposures'].dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
