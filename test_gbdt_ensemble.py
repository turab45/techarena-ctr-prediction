import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import gbdt


class PositionFeatureTests(unittest.TestCase):
    def test_position_preserves_candidate_order_and_similarity_relative_features(self):
        rng = np.random.default_rng(11)
        ds = dict(items=np.tile(np.arange(10), (2, 1)), user=np.array([0, 1]),
                  hist=np.full((2, 20), -1), tb=np.zeros(2, dtype=int),
                  dt=np.zeros(2, dtype=int), ts=np.array([100, 200]))
        ds['hist'][1, :2] = [2, 3]
        emb = rng.normal(size=(10, 32)).astype(np.float32)
        emb64 = rng.normal(size=(10, 64)).astype(np.float32)
        stats = ((.1, np.ones(2), np.ones(2), np.ones(10), np.ones(10), emb),
                 emb64, rng.normal(size=(2, 64)).astype(np.float32))
        x = gbdt._feature_core(ds, np.array([0, 1]), stats).reshape(2, 10, 53)
        np.testing.assert_array_equal(x[:, :, 13], np.tile(np.arange(10), (2, 1)))
        # The original most-recent-history similarity is now column 14.
        expected = emb[3] @ emb.T
        np.testing.assert_allclose(x[1, :, 14], expected, rtol=1e-5)
        np.testing.assert_allclose(x[1, :, 35], expected - expected.mean(), rtol=1e-5)
        np.testing.assert_allclose(x[1, :, 36], expected - expected.max(), atol=1e-5)
        self.assertTrue(np.isfinite(x).all())

    def test_full_block_dimensions_component_boundaries_and_cache_rejection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(gbdt, 'FEAT', Path(directory)):
            path = gbdt.FEAT / f'block_test_c100_v{gbdt.FEATURE_VERSION}.npy'
            np.save(path, np.zeros((20, 354), dtype=np.float32))
            with patch.object(gbdt, 'get_preference', return_value=None), \
                    patch.object(gbdt, 'get_context_counts', return_value=None), \
                    patch.object(gbdt, '_feature_preference', return_value=np.ones((20, 191))), \
                    patch.object(gbdt, '_warm_component_features', return_value=np.full((20, 68), 2)), \
                    patch.object(gbdt, '_feature_context_prior', return_value=np.full((20, 32), 3)), \
                    patch.object(gbdt, '_als_component_features', return_value=np.full((20, 64), 4)):
                x = gbdt.build_block({}, {}, 2, 10, 0, 'test', np.array([0, 1]), 100)
            self.assertEqual(x.shape, (20, 355))
            self.assertEqual(x.dtype, np.float32)
            for section, value in ((slice(0, 191), 1), (slice(191, 259), 2),
                                   (slice(259, 291), 3), (slice(291, 355), 4)):
                self.assertTrue((x[:, section] == value).all())
            np.testing.assert_array_equal(gbdt.build_block({}, {}, 2, 10, 0, 'test',
                                                         np.array([0, 1]), 100), x)

    def test_assembled_cache_rejects_old_width_even_when_row_count_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'matrix.npy'
            np.save(path, np.zeros((10, 354), dtype=np.float32))
            np.save(path.with_name('matrix_labels.npy'), np.zeros(10, dtype=np.int8))
            blocks = [dict(events=np.array([0]), nu=1, ni=10, start=0, name='test', cutoff=100)]
            with patch.object(gbdt, 'build_block', return_value=np.ones((10, 355))) as build:
                x, y = gbdt._assemble(dict(labels=np.zeros((1, 10))), blocks, path)
            build.assert_called_once()
            self.assertEqual(x.shape, (10, 355))


class EnsembleTests(unittest.TestCase):
    def test_validation_and_final_average_seeds_and_preserve_individual_tree_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            x = np.zeros((2, 355))
            y = np.array([0, 1])
            models = [MagicMock(), MagicMock()]
            models[0].predict.return_value = np.array([.1, .9])
            models[1].predict.return_value = np.array([.3, .7])
            with patch.object(gbdt, 'train_lightgbm', side_effect=[(models[0], 7), (models[1], 11)]) as train:
                scores, counts, metrics = gbdt.train_ensemble(x, y, x, y, [42, 43], output, 2)
                np.testing.assert_allclose(scores, [.2, .8])
                self.assertEqual(counts, [7, 11])
                self.assertEqual(metrics, {'42': 1., '43': 1.})
                self.assertEqual([call.kwargs['seed'] for call in train.call_args_list], [42, 43])
            evaluation = np.ones((2, 355))
            for model in models:
                model.reset_mock()
            with patch.object(gbdt, 'train_lightgbm', side_effect=[(models[0], 7), (models[1], 11)]) as train:
                scores, _, _ = gbdt.train_ensemble(x, y, x, y, [42, 43], output, 2,
                                                  rounds=counts, prediction_x=evaluation)
                self.assertEqual([call.kwargs['num_boost_round'] for call in train.call_args_list], [7, 11])
                self.assertTrue(all(call.kwargs['early_stopping'] == 0 for call in train.call_args_list))
                for model in models:
                    self.assertIs(model.predict.call_args.args[0], evaluation)
                    model.save_model.assert_called_once()
                np.testing.assert_allclose(scores, [.2, .8])


if __name__ == '__main__':
    unittest.main()
