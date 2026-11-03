import io
import pickle
import tempfile
import unittest
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import pipeline


class PretrainedTrainingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cache = patch.object(pipeline, 'CACHE', Path(tmp.name))
        cache.start()
        self.addCleanup(cache.stop)
        self.embeddings = {
            'known': np.arange(4, dtype=np.float32) / 10,
            'long': np.arange(12, dtype=np.float32) / 10,
            'outside': np.ones(4, dtype=np.float32),
        }
        self.write_embeddings()
        events = pd.DataFrame([
            dict(items=['known', 'long'], labels=[1, 0], history_items=[],
                 history_timestamps=[], user_id=1, timestamp=7200,
                 time_bucket=0, day_type=0),
            dict(items=['known', 'long'], labels=[0, 1], history_items=['known'],
                 history_timestamps=[3600], user_id=2, timestamp=7200,
                 time_bucket=1, day_type=1),
        ])
        self.events = events
        self.ds = pipeline.SlateDataset(
            events, {'known': 2, 'long': 1, 'missing': 3}, {1: 1, 2: 2},
            max_hist=3, slate_size=2)
        self.dl = DataLoader(self.ds, batch_size=2)
        self.cfg = pipeline.ModelCfg(
            num_items=4, num_users=3, num_tb=3, num_dt=3,
            d_model=8, n_heads=2, dropout=0, max_hist=3, slate_size=2)

    def write_embeddings(self):
        with (pipeline.CACHE/'item2vec.pkl').open('wb') as f:
            pickle.dump(self.embeddings, f)

    def test_embeddings_follow_vocab_indices_and_preserve_padding(self):
        torch.manual_seed(7)
        baseline = pipeline.Model(self.cfg)
        torch.manual_seed(7)
        with patch('sys.stdout', new_callable=io.StringIO) as output:
            model = pipeline.Model(self.cfg, self.ds.item2idx)
        weights = model.item_emb.weight.detach()
        torch.testing.assert_close(weights[2, :4], torch.from_numpy(self.embeddings['known']))
        torch.testing.assert_close(weights[2, 4:], baseline.item_emb.weight[2, 4:])
        torch.testing.assert_close(weights[1], torch.from_numpy(self.embeddings['long'][:8]))
        torch.testing.assert_close(weights[3], baseline.item_emb.weight[3])
        torch.testing.assert_close(weights[0], torch.zeros(8))
        self.assertIn('loaded 2 / 3 pretrained item embeddings', output.getvalue())

    def test_unmatched_pretrained_ids_fail(self):
        self.embeddings = {'outside': np.ones(8, dtype=np.float32)}
        self.write_embeddings()
        with self.assertRaisesRegex(ValueError, 'No pretrained item IDs match'):
            pipeline.Model(self.cfg, self.ds.item2idx)

    def test_nonfinite_pretrained_vectors_fail(self):
        self.embeddings['known'][0] = np.nan
        self.write_embeddings()
        with self.assertRaisesRegex(ValueError, 'Invalid pretrained embedding'):
            pipeline.Model(self.cfg, self.ds.item2idx)

    def test_prediction_restores_checkpoint_without_item2vec_file(self):
        model = pipeline.Model(self.cfg, self.ds.item2idx).eval()
        checkpoint = io.BytesIO()
        torch.save({'model': model.state_dict(), 'cfg': asdict(self.cfg)}, checkpoint)
        (pipeline.CACHE/'item2vec.pkl').unlink()
        checkpoint.seek(0)
        saved = torch.load(checkpoint, weights_only=True)
        restored = pipeline.Model(pipeline.ModelCfg(**saved['cfg'])).eval()
        restored.load_state_dict(saved['model'])
        batch = next(iter(self.dl))
        with torch.no_grad():
            torch.testing.assert_close(pipeline.forward(model, batch),
                                       pipeline.forward(restored, batch))

    def test_missing_item2vec_stops_before_loading_datasets(self):
        (pipeline.CACHE/'item2vec.pkl').unlink()
        with patch.object(pipeline, 'make_loaders') as loaders:
            with self.assertRaisesRegex(SystemExit, 'pretrain_item2vec.py'):
                pipeline.cmd_train(Namespace(device='cpu'))
        loaders.assert_not_called()

    def test_new_cli_and_model_defaults(self):
        with patch('sys.argv', ['pipeline.py', 'train']), \
                patch.object(pipeline, 'cmd_train') as train:
            pipeline.main()
        args = train.call_args.args[0]
        self.assertEqual((args.device, args.epochs, args.lr, args.d_model),
                         ('cuda', 15, 1e-3, 64))
        self.assertEqual(args.dropout, 0.10)
        cfg = pipeline.ModelCfg(num_items=4, num_users=3, num_tb=3, num_dt=3)
        self.assertEqual((cfg.d_model, cfg.hist_layers, cfg.slate_layers, cfg.d_ff),
                         (64, 2, 1, 256))
        self.assertEqual(cfg.dropout, 0.10)
        self.assertTrue(cfg.numeric_features)
        self.assertTrue(cfg.history_pooling)

    def test_numeric_statistics_and_scaling(self):
        items, users, global_ctr = pipeline.compute_stats_from_events(self.events)
        self.assertEqual(global_ctr, 0.5)
        self.assertEqual(items['known'], (0.5, 2, 7200))
        self.assertEqual(users[1], (1, 1))
        ds = pipeline.SlateDataset(
            self.events, self.ds.item2idx, {1: 1, 2: 2}, max_hist=3, slate_size=2,
            item_stats=items, user_stats=users, global_ctr=global_ctr)
        row = ds[1]
        self.assertAlmostEqual(row['item_ctr'][0].item(), 0.5)
        self.assertAlmostEqual(row['item_pop'][0].item(), np.log1p(2) / 15)
        self.assertNotIn('item_age', row)
        self.assertAlmostEqual(row['user_act'].item(), np.log1p(1) / 5)
        self.assertAlmostEqual(row['user_ctr'].item(), 0.3)
        self.assertAlmostEqual(row['in_hist'][0].item(), 1 / 20)
        self.assertAlmostEqual(row['hour_sin'].item(), 0.5)
        self.assertAlmostEqual(row['hour_cos'].item(), np.sqrt(3) / 2)

    def test_numeric_features_reach_the_head(self):
        model = pipeline.Model(self.cfg).eval()
        batch = next(iter(self.dl))
        captured = []
        hook = model.head[0].register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach()))
        try:
            with torch.no_grad():
                logits = pipeline.forward(model, batch)
        finally:
            hook.remove()
        self.assertEqual(logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits).all().item())
        expected_cand = torch.stack([batch[k] for k in
                                    ('in_hist', 'item_ctr', 'item_pop')], dim=-1)
        expected_event = torch.stack([batch[k] for k in
                                     ('user_act', 'user_ctr', 'hour_sin', 'hour_cos',
                                      'dow_sin', 'dow_cos')], dim=-1)
        offset = self.cfg.d_model * 5
        self.assertEqual(captured[0].shape[-1], offset + 9)
        torch.testing.assert_close(captured[0][..., offset:offset + 3], expected_cand)
        torch.testing.assert_close(captured[0][..., offset + 3:], expected_event[:, None, :].expand(-1, 2, -1))

    def test_history_pools_ignore_padding_and_select_last_valid_position(self):
        model = pipeline.Model(self.cfg).eval()
        encoded = torch.arange(72, dtype=torch.float32).reshape(3, 3, 8)
        encoded[0, 1:] = 1000
        encoded[1, 2] = 2000
        hist = torch.zeros(3, 3, dtype=torch.long)
        with patch.object(model.hist_enc, 'forward', return_value=encoded):
            x, pad, mean, maximum, last = model.encode_history(
                hist, torch.tensor([0, 2, 3]), torch.zeros(3, 3))
        self.assertEqual(pad.tolist(), [[False, True, True],
                                       [False, False, True], [False, False, False]])
        for row, length in enumerate((1, 2, 3)):
            torch.testing.assert_close(mean[row], x[row, :length].mean(0))
            torch.testing.assert_close(maximum[row], x[row, :length].max(0).values)
            torch.testing.assert_close(last[row], x[row, length - 1])
        self.assertTrue(torch.isfinite(mean).all().item())
        self.assertTrue(torch.isfinite(maximum).all().item())

    def test_pooling_head_without_numeric_features(self):
        cfg = pipeline.ModelCfg(**{**asdict(self.cfg), 'numeric_features': False})
        model = pipeline.Model(cfg).eval()
        self.assertEqual(model.head[0].in_features, cfg.d_model * 5 + 1)
        with torch.no_grad():
            logits = pipeline.forward(model, next(iter(self.dl)))
        self.assertEqual(logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_stats_require_aligned_labeled_events(self):
        with self.assertRaisesRegex(ValueError, 'Need labeled events'):
            pipeline.compute_stats_from_events(self.events.drop(columns='labels'))
        invalid = self.events.copy()
        invalid.at[0, 'labels'] = [1]
        with self.assertRaisesRegex(ValueError, 'matching lengths'):
            pipeline.compute_stats_from_events(invalid)

    def check_training(self, device, auc_values=None):
        args = Namespace(device=device, d_model=8, n_heads=2, dropout=0,
                         max_hist=3, slate_size=2, lr=1e-3, epochs=3)
        if auc_values is not None:
            args.epochs = len(auc_values)
        observations, schedulers, updates = [], [], []
        starting = {}
        original_forward = pipeline.forward
        original_scheduler = torch.optim.lr_scheduler.LambdaLR

        def forward(model, batch):
            self.assertEqual(next(model.parameters()).device.type, device)
            self.assertTrue(all(t.device.type == device for t in batch.values()))
            if not starting:
                starting['items'] = model.item_emb.weight.detach().cpu().clone()
                starting['head'] = model.head[-1].weight.detach().cpu().clone()
            return original_forward(model, batch)

        def evaluate(model, dl, dev):
            observations.append((model.item_emb.weight.requires_grad,
                                 model.item_emb.weight.detach().cpu().clone(),
                                 model.head[-1].weight.detach().cpu().clone()))
            return (auc_values[len(observations) - 1] if auc_values is not None
                    else 0.6 + len(observations) / 100)

        def scheduler(opt, fn):
            self.assertEqual(opt.defaults['weight_decay'], 1e-5)
            opt.register_step_post_hook(lambda *args: updates.append(1))
            result = original_scheduler(opt, fn)
            schedulers.append(result)
            return result

        def autocast(device_type, dtype):
            return torch.amp.autocast(device_type, dtype=dtype, enabled=device_type == 'cuda')

        scaler = torch.amp.GradScaler(device, enabled=device == 'cuda')
        batches = [next(iter(self.dl)) for _ in range(8)]
        with patch.object(pipeline, 'make_loaders', return_value=(self.ds, batches, self.dl)), \
                patch.object(pipeline, 'forward', side_effect=forward), \
                patch.object(pipeline, 'evaluate', side_effect=evaluate), \
                patch.object(pipeline, 'autocast', side_effect=autocast), \
                patch.object(pipeline, 'GradScaler', return_value=scaler), \
                patch.object(torch.optim.lr_scheduler, 'LambdaLR', side_effect=scheduler), \
                patch.object(torch, 'save') as save, \
                patch('sys.stdout', new_callable=io.StringIO) as output:
            pipeline.cmd_train(args)
        self.assertEqual([o[0] for o in observations],
                         [False, False] + [True] * (len(observations) - 2))
        torch.testing.assert_close(observations[0][1], starting['items'], rtol=0, atol=0)
        torch.testing.assert_close(observations[1][1], starting['items'], rtol=0, atol=0)
        self.assertFalse(torch.equal(observations[2][1], starting['items']))
        self.assertFalse(torch.equal(observations[0][2], starting['head']))
        self.assertGreater(len(updates), 0)
        self.assertEqual(schedulers[0].last_epoch, len(updates))
        self.assertAlmostEqual(schedulers[0].get_last_lr()[0], args.lr * len(updates) / 500)
        best_saves = sum(call.args[1] == 'best_model.pt' for call in save.call_args_list)
        epoch_saves = [call for call in save.call_args_list
                       if str(call.args[1]).startswith('ckpts/')]
        self.assertEqual(len(epoch_saves), len(observations))
        for epoch, call in enumerate(epoch_saves, 1):
            expected_auc = auc_values[epoch - 1] if auc_values is not None else 0.6 + epoch / 100
            self.assertEqual(call.args[1], f'ckpts/epoch_{epoch:02d}_auc_{expected_auc:.5f}.pt')
            self.assertEqual(call.args[0]['epoch'], epoch)
            self.assertEqual(call.args[0]['auc'], expected_auc)
        if auc_values is None:
            self.assertEqual(best_saves, 3)
        self.assertIn('[unfreeze]', output.getvalue())
        return observations, best_saves, output.getvalue()

    def test_early_stopping_preserves_best_checkpoint(self):
        observations, saves, output = self.check_training('cpu', [0.6, 0.7, 0.69, 0.68, 0.67, 0.66, 0.9])
        self.assertEqual(len(observations), 6)
        self.assertEqual(saves, 2)
        self.assertIn('[early-stop] no improvement for 4 epochs', output)
        self.assertIn('[done] best val AUC = 0.70000', output)

    def test_zero_auc_still_saves_the_first_checkpoint(self):
        observations, saves, output = self.check_training('cpu', [0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        self.assertEqual(len(observations), 5)
        self.assertEqual(saves, 1)
        self.assertIn('[done] best val AUC = 0.00000', output)

    def test_loaders_only_enable_augmentation_for_training(self):
        events = self.events.copy()
        events['timestamp'] = [7200, 14400]
        events.to_pickle(pipeline.CACHE/'train.pkl')
        with (pipeline.CACHE/'vocab.pkl').open('wb') as f:
            pickle.dump({'item2idx': self.ds.item2idx, 'user2idx': {1: 1, 2: 2}}, f)
        args = Namespace(val_frac=0.15, max_hist=3, slate_size=2, bs=1, workers=0)
        ds_tr, _, dl_va = pipeline.make_loaders(args)
        self.assertTrue(ds_tr.training)
        self.assertFalse(dl_va.dataset.training)
        self.assertFalse(self.ds.training)
        before = dl_va.dataset.item_ctr.copy()
        events.at[1, 'labels'] = [1, 0]
        events.to_pickle(pipeline.CACHE/'train.pkl')
        _, _, changed_val = pipeline.make_loaders(args)
        np.testing.assert_array_equal(before, changed_val.dataset.item_ctr)

    def test_statistics_cache_is_written_and_reused(self):
        events = self.events.copy()
        events['timestamp'] = [7200, 14400]
        events.to_pickle(pipeline.CACHE/'train.pkl')
        with (pipeline.CACHE/'vocab.pkl').open('wb') as f:
            pickle.dump({'item2idx': self.ds.item2idx, 'user2idx': {1: 1, 2: 2}}, f)
        args = Namespace(val_frac=0.15, max_hist=3, slate_size=2, bs=1, workers=0)
        with patch.object(pipeline, 'compute_stats_from_events',
                          wraps=pipeline.compute_stats_from_events) as compute:
            first, _, _ = pipeline.make_loaders(args)
            second, _, _ = pipeline.make_loaders(args)
        compute.assert_called_once()
        self.assertEqual(len(list(pipeline.CACHE.glob('stats_*.pkl'))), 1)
        np.testing.assert_array_equal(first.item_ctr, second.item_ctr)
        np.testing.assert_array_equal(first.user_ctr, second.user_ctr)

    def test_training_freeze_unfreeze_and_per_update_warmup(self):
        self.check_training('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_pretrained_training_on_gpu(self):
        self.check_training('cuda')


class HistoryPreservationTests(unittest.TestCase):
    def setUp(self):
        events = pd.DataFrame([dict(
            items=[3, 8], labels=[1, 0], history_items=list(range(1, 9)),
            history_timestamps=list(range(0, 28800, 3600)), user_id=1,
            timestamp=28800, time_bucket=0, day_type=0)])
        self.ds = pipeline.SlateDataset(events, {i: i for i in range(1, 9)}, {1: 1},
                                        max_hist=10, slate_size=2)

    def test_all_modes_preserve_complete_history_and_aligned_gaps(self):
        for training in (True, False):
            with self.subTest(training=training):
                self.ds.training = training
                with patch.object(np.random, 'rand', side_effect=AssertionError('Unexpected augmentation')):
                    row = self.ds[0]
                self.assertEqual(row['hist_len'].item(), 8)
                self.assertEqual(row['hist'].tolist(), list(range(1, 9)) + [0, 0])
                self.assertEqual(row['hist_gap'].tolist(), list(range(8, 0, -1)) + [0, 0])


if __name__ == '__main__':
    unittest.main()
