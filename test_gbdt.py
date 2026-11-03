import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import gbdt


class Item2VecFeaturesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cache = patch.object(gbdt, 'CACHE', Path(directory.name))
        cache.start()
        self.addCleanup(cache.stop)
        with (gbdt.CACHE/'item2vec.pkl').open('wb') as f:
            pickle.dump({'a': np.pad(np.array([1., 0.]), (0, 62)),
                         'b': np.pad(np.array([0., 2.]), (0, 62)),
                         'c': np.pad(np.array([-1., 0.]), (0, 62))}, f)

    def test_cosine_pools_preserve_rows_and_handle_missing_embeddings(self):
        events = pd.DataFrame({'history_items': [['a', 'b', 'unknown'], [], ['unknown'], ['c']]})
        flat = pd.DataFrame({'event_row': [0, 0, 1, 2, 3, 0],
                             'item_id': ['a', 'c', 'a', 'a', 'a', 'a']},
                            index=[12, 4, 7, 3, 9, 1])
        original = flat.copy()
        result = gbdt.add_item2vec_features(flat, events)
        pd.testing.assert_frame_equal(result[['event_row', 'item_id']], original)
        np.testing.assert_allclose(
            result[['item2vec_sim_max', 'item2vec_sim_mean', 'item2vec_sim_last']],
            [[1, 0.5, 0], [0, -0.5, 0], [0, 0, 0], [0, 0, 0],
             [-1, -1, -1], [1, 0.5, 0]])
        self.assertTrue(all(result[col].dtype == np.float32 for col in
                            ('item2vec_sim_max', 'item2vec_sim_mean', 'item2vec_sim_last')))
        np.testing.assert_allclose(result['item2vec_sim_user_profile'],
                                   [0.5, -0.5, 0, 0, -1, 0.5])
        emb_cols = [f'item2vec_dim_{i}' for i in range(64)]
        self.assertTrue(all(result[col].dtype == np.float32 for col in emb_cols))
        np.testing.assert_allclose(result[emb_cols].iloc[1].to_numpy(),
                                   np.pad(np.array([-1., 0.]), (0, 62)))

    def test_empty_histories_and_unknown_candidates_return_zero(self):
        events = pd.DataFrame({'history_items': [[]]})
        flat = pd.DataFrame({'event_row': [0], 'item_id': ['unknown']})
        result = gbdt.add_item2vec_features(flat, events)
        np.testing.assert_array_equal(
            result[['item2vec_sim_max', 'item2vec_sim_mean', 'item2vec_sim_last']],
            np.zeros((1, 3)))
        self.assertEqual(result['item2vec_sim_user_profile'].iloc[0], 0)
        np.testing.assert_array_equal(
            result[[f'item2vec_dim_{i}' for i in range(64)]], np.zeros((1, 64)))


if __name__ == '__main__':
    unittest.main()
