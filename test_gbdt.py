import pickle
from argparse import Namespace
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
        self.assertEqual(result['item2vec_sim_last_3'].iloc[0], 0)
        self.assertEqual(result['item2vec_sim_weighted'].iloc[0], 0)
        np.testing.assert_array_equal(
            result[[f'item2vec_dim_{i}' for i in range(64)]], np.zeros((1, 64)))

    def test_last_three_and_weighted_similarity_favor_recent_items(self):
        events = pd.DataFrame({'history_items': [['a', 'b', 'b', 'c']]})
        flat = pd.DataFrame({'event_row': [0, 0], 'item_id': ['a', 'unknown']})
        result = gbdt.add_item2vec_features(flat, events)
        # Cosines are [1, 0, 0, -1]; the newest, opposite vector gets weight 1.
        expected_weighted = (1 / 4 - 1) / (1 / 4 + 1 / 3 + 1 / 2 + 1)
        self.assertAlmostEqual(result['item2vec_sim_last_3'].iloc[0], -1 / 3)
        self.assertAlmostEqual(result['item2vec_sim_weighted'].iloc[0], expected_weighted)
        self.assertLess(result['item2vec_sim_weighted'].iloc[0], 0)
        self.assertEqual(result['item2vec_sim_last_3'].iloc[1], 0)
        self.assertEqual(result['item2vec_sim_weighted'].iloc[1], 0)


class FeatureCacheTests(unittest.TestCase):
    def test_cache_keys_track_split_sample_als_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(gbdt, 'CACHE', Path(directory)), \
                patch.object(gbdt, 'HAS_IMPLICIT', True):
            for name in ('train.pkl', 'eval.pkl', 'item2vec.pkl', 'vocab.pkl'):
                (gbdt.CACHE/name).write_bytes(b'cache input')
            args = Namespace(val_frac=0.15, sample=0, skip_als=False)
            original = gbdt.feature_cache_paths(args)
            self.assertEqual(original, gbdt.feature_cache_paths(args))
            for changes in ({'val_frac': 0.1}, {'sample': 50000}, {'skip_als': True}):
                changed_args = Namespace(**{**vars(args), **changes})
                changed = gbdt.feature_cache_paths(changed_args)
                self.assertTrue(all(old != new for old, new in zip(original, changed)))
            (gbdt.CACHE/'vocab.pkl').write_bytes(b'updated vocabulary input')
            changed = gbdt.feature_cache_paths(args)
            self.assertTrue(all(old != new for old, new in zip(original, changed)))


class LightGBMTrainingTests(unittest.TestCase):
    def test_training_preserves_categorical_ids_and_regularization(self):
        data = pd.DataFrame({'user_idx': [1.0, 2.0], 'item_idx': [3.0, 4.0],
                             'numeric': [0.2, 0.8], 'label': [0, 1]})
        for columns in (['user_idx', 'item_idx', 'numeric'], ['numeric']):
            with self.subTest(columns=columns), \
                    patch.object(gbdt.lgb, 'Dataset') as dataset, \
                    patch.object(gbdt.lgb, 'train') as train:
                train.return_value.best_iteration = 3
                train.return_value.predict.return_value = np.array([0.2, 0.8])
                train.return_value.feature_importance.return_value = np.ones(len(columns))
                model, auc = gbdt.train_lgb(data.copy(), data.copy(), columns, 5)
                self.assertIs(model, train.return_value)
                self.assertEqual(auc, 1.0)
                params = train.call_args.args[0]
                for key, value in {'learning_rate': 0.02, 'num_leaves': 63,
                                   'min_data_in_leaf': 500, 'lambda_l2': 10.0,
                                   'feature_fraction': 0.6, 'bagging_fraction': 0.7,
                                   'cat_smooth': 20, 'cat_l2': 10}.items():
                    self.assertEqual(params[key], value)
                self.assertEqual(train.call_args.kwargs['num_boost_round'], 5)
                categorical = [c for c in ('user_idx', 'item_idx') if c in columns]
                self.assertEqual(dataset.call_args_list[0].kwargs['categorical_feature'], categorical)
                for call in dataset.call_args_list:
                    self.assertEqual(list(call.args[0].columns), columns)
                    for col in categorical:
                        self.assertIsInstance(call.args[0][col].dtype, pd.CategoricalDtype)
                self.assertNotIn('user_item_ctr', gbdt.FEATURE_COLS)


class ALSHistoryTests(unittest.TestCase):
    def test_clicks_and_pre_cutoff_history_form_binary_interactions(self):
        flat = pd.DataFrame({'user_id': ['u', 'u', 'v', 'v'],
                             'item_id': ['a', 'b', 'a', 'c'], 'label': [1, 0, 0, 1]})
        events = pd.DataFrame({
            'user_id': ['u', 'v'],
            'history_items': [['a', 'c', 'b'], ['a', 'a', 'unknown']],
            'history_timestamps': [[0, 50, 100], [20, 20, 0]],
        })
        with patch.object(gbdt, 'HAS_IMPLICIT', True), \
                patch.object(gbdt, 'AlternatingLeastSquares', create=True) as als:
            als.return_value.user_factors = np.ones((2, 4), dtype=np.float32)
            als.return_value.item_factors = np.ones((3, 4), dtype=np.float32)
            _, users, items = gbdt.train_als(flat, events, 100, factors=4)
        matrix = als.return_value.fit.call_args.args[0].toarray()
        self.assertEqual(matrix[users['u'], items['a']], 1)
        self.assertEqual(matrix[users['u'], items['c']], 1)
        self.assertEqual(matrix[users['u'], items['b']], 0)
        self.assertEqual(matrix[users['v'], items['a']], 1)
        self.assertEqual(matrix[users['v'], items['c']], 1)
        self.assertTrue(np.all((matrix == 0) | (matrix == 1)))


if __name__ == '__main__':
    unittest.main()
