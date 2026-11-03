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
                         ('cuda', 15, 3e-4, 64))
        self.assertEqual(args.dropout, 0.25)
        cfg = pipeline.ModelCfg(num_items=4, num_users=3, num_tb=3, num_dt=3)
        self.assertEqual((cfg.d_model, cfg.hist_layers, cfg.slate_layers, cfg.d_ff),
                         (64, 1, 1, 256))
        self.assertEqual(cfg.dropout, 0.25)

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
            self.assertEqual(opt.defaults['weight_decay'], 1e-3)
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
        observations, saves, output = self.check_training('cpu', [0.6, 0.7, 0.69, 0.68, 0.67, 0.9])
        self.assertEqual(len(observations), 5)
        self.assertEqual(saves, 2)
        self.assertIn('[early-stop] no improvement for 3 epochs', output)
        self.assertIn('[done] best val AUC = 0.70000', output)

    def test_zero_auc_still_saves_the_first_checkpoint(self):
        observations, saves, output = self.check_training('cpu', [0.0, 0.0, 0.0, 0.0, 0.5])
        self.assertEqual(len(observations), 4)
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

    def test_training_freeze_unfreeze_and_per_update_warmup(self):
        self.check_training('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_pretrained_training_on_gpu(self):
        self.check_training('cuda')


class HistoryDropoutTests(unittest.TestCase):
    def setUp(self):
        events = pd.DataFrame([dict(
            items=[3, 8], labels=[1, 0], history_items=list(range(1, 9)),
            history_timestamps=list(range(0, 28800, 3600)), user_id=1,
            timestamp=28800, time_bucket=0, day_type=0)])
        self.ds = pipeline.SlateDataset(events, {i: i for i in range(1, 9)}, {1: 1},
                                        max_hist=10, slate_size=2)

    def test_training_dropout_keeps_items_and_gaps_aligned_without_mutation(self):
        self.ds.training = True
        draws = np.array([0.01, 0.9, 0.02, 0.8, 0.1, 0.99, 0.05, 0.0])
        with patch.object(np.random, 'rand', return_value=draws):
            row = self.ds[0]
        self.assertEqual(row['hist_len'].item(), 4)
        self.assertEqual(row['hist'].tolist(), [2, 4, 6, 8, 0, 0, 0, 0, 0, 0])
        self.assertEqual(row['hist_gap'].tolist(), [7, 5, 3, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(self.ds.hist[0].tolist(), list(range(1, 9)) + [0, 0])
        self.assertEqual(self.ds.hist_gap[0].tolist(), list(range(8, 0, -1)) + [0, 0])

    def test_evaluation_and_prediction_do_not_drop_history(self):
        with patch.object(np.random, 'rand', side_effect=AssertionError('Unexpected augmentation')):
            row = self.ds[0]
        self.assertEqual(row['hist_len'].item(), 8)
        self.assertEqual(row['hist'].tolist(), list(range(1, 9)) + [0, 0])

    def test_training_always_preserves_the_latest_item(self):
        self.ds.training = True
        with patch.object(np.random, 'rand', return_value=np.zeros(8)):
            row = self.ds[0]
        self.assertEqual(row['hist_len'].item(), 1)
        self.assertEqual(row['hist'].tolist(), [8] + [0] * 9)
        self.assertEqual(row['hist_gap'].tolist(), [1] + [0] * 9)


if __name__ == '__main__':
    unittest.main()
