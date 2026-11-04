#!/usr/bin/env python3
"""OFFLINE TechArena CTR: GPU CatBoost on all 355 features with history-shift weighting.

Examples (NVIDIA CUDA machine):
    uv run python final_sprint_history_gpu.py --device gpu --trials uniform,hist_mild,hist_full
    uv run python final_sprint_history_gpu.py --device gpu --trials uniform,hist_mild --backtest-day5

Uses saved gbdt v5 matrices. Does not write a submission or train on evaluation labels.
The evaluation feature history distribution is used only to propose unsupervised weights.
Requires final_sprint_ablation_v2.py from the previous experiment alongside this file.
"""
from __future__ import annotations
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from final_sprint_ablation_v2 import load_xy, contexts, _row_ids, report_metrics

BINS = [0, 3, 7, 15]

def group_hist_len(mat):
    hist_len = np.asarray(mat[::10, 6], dtype=np.float64)
    if not np.isfinite(hist_len).all() or not np.all((hist_len >= 0) & (hist_len <= 20)):
        raise ValueError('Unexpected history length feature in column 6')
    return np.digitize(hist_len.astype(np.int16), BINS, right=True)

def ratio_from_counts(source_bins, target_bins, max_ratio=4.0):
    src = np.bincount(source_bins, minlength=5).astype(np.float64) + 1
    tgt = np.bincount(target_bins, minlength=5).astype(np.float64) + 1
    raw = (tgt / tgt.sum()) / (src / src.sum())
    return np.clip(raw, 0.25, max_ratio)

def sample_weights(train_bins, ratio, mode, recent_event_weights):
    if mode == 'uniform':
        w = np.ones(len(train_bins), np.float64)
    elif mode == 'hist_mild':
        w = np.sqrt(ratio[train_bins])
    elif mode == 'hist_full':
        w = ratio[train_bins]
    elif mode == 'hist_mild_recent':
        w = np.sqrt(ratio[train_bins]) * recent_event_weights
    else:
        raise ValueError(mode)
    w = np.clip(w, 0.2, 5.0)
    w = w / w.mean()
    return np.repeat(w.astype(np.float32), 10)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, default=Path('cache/features_gbdt_v5'))
    ap.add_argument('--meta', type=Path, default=Path('cache/gbdt_data.npz'))
    ap.add_argument('--out', type=Path, default=Path('artifacts/history_gpu_sprint'))
    # CatBoost GPU support requires a compatible CUDA runtime. Default to CPU
    # so this script runs on Macs and other machines without CatBoost GPU.
    ap.add_argument('--device', choices=['cpu','gpu'], default='cpu')
    ap.add_argument('--gpu-id', type=int, default=0)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--trials', default='uniform,hist_mild,hist_full')
    ap.add_argument('--backtest-day5', action='store_true')
    ap.add_argument('--iterations', type=int, default=900)
    ap.add_argument('--patience', type=int, default=75)
    ap.add_argument('--depth', type=int, default=7)
    ap.add_argument('--learning-rate', type=float, default=0.05)
    ap.add_argument('--l2', type=float, default=12.0)
    args = ap.parse_args()
    # CatBoost's GPU backend requires NVIDIA CUDA. macOS (including Apple
    # Silicon) does not provide that backend, so transparently use CPU even
    # when a copied command includes ``--device gpu``.
    if args.device == 'gpu' and platform.system() == 'Darwin':
        print('[device] macOS detected; CatBoost GPU requires NVIDIA CUDA. Using CPU.', flush=True)
        args.device = 'cpu'
    trials = [x.strip() for x in args.trials.split(',') if x.strip()]
    if not trials or any(x not in {'uniform','hist_mild','hist_full','hist_mild_recent'} for x in trials):
        ap.error('Valid trials: uniform,hist_mild,hist_full,hist_mild_recent')
    if args.iterations < 2 or args.patience < 1:
        ap.error('Bad iterations or patience')
    x, y = load_xy(args.cache, 'train')
    xv, yv = load_xy(args.cache, 'val')
    ctx = contexts(args.meta, x, y, xv, yv)
    if ctx is None or ctx['ev_hlen'] is None:
        ap.error('Need cache/gbdt_data.npz with evaluation history metadata')
    if args.backtest_day5:
        split = sum(ctx['train_counts'][:2]) * 10
        if not 0 < split < len(y):
            ap.error('Invalid day5 split')
        # Source: days 3-4, target (unlabelled distribution only): day 5.
        target_bins = group_hist_len(x[split:])
        xv, yv = x[split:], y[split:]
        x, y = x[:split], y[:split]
        ctx = {**ctx, 'first': ctx['day5_first'], 'late': ctx['day5_late'],
               'train_counts': ctx['train_counts'][:2]}
        args.out = args.out / 'day5_backtest'
    else:
        # Distribution only, no evaluation labels or evaluation predictions.
        target_bins = np.digitize(np.asarray(ctx['ev_hlen'], dtype=np.int16), BINS, right=True)
    train_bins = group_hist_len(x)
    val_bins = group_hist_len(xv)
    ratio = ratio_from_counts(train_bins, target_bins)
    val_ratio = ratio_from_counts(val_bins, target_bins)
    val_w = np.repeat(val_ratio[val_bins], 10)
    if ctx['train_counts']:
        day_vals = np.array([0.6,1.0,1.5] if len(ctx['train_counts']) == 3 else [0.75,1.25], np.float64)
        recent = np.repeat(day_vals, ctx['train_counts'])
    else:
        recent = np.ones(len(train_bins), np.float64)
    if len(recent) != len(train_bins):
        raise SystemExit('STOP: day/event counts do not line up with training matrix')
    print('[history] train group counts:', np.bincount(train_bins,minlength=5), flush=True)
    print('[history] target group counts:', np.bincount(target_bins,minlength=5), flush=True)
    print('[history] train-to-target weight ratios:', np.round(ratio,3), flush=True)
    print('[history] validation-to-target weight ratios:', np.round(val_ratio,3), flush=True)
    print(f'[strategy] 355 columns, {args.device.upper()} CatBoost, no submission', flush=True)
    # A full materialization of 355 float32 columns is ~4.1 GB for 2.9M candidate rows.
    # Avoid copying both matrices more than once.
    xt = np.asarray(x, dtype=np.float32)
    va = np.asarray(xv, dtype=np.float32)
    if not np.isfinite(xt).all() or not np.isfinite(va).all():
        raise SystemExit('STOP: nonfinite features')
    if not np.isin(y,[0,1]).all() or not np.isin(yv,[0,1]).all():
        raise SystemExit('STOP: nonbinary labels')
    args.out.mkdir(parents=True,exist_ok=True)
    outputs = []
    for mode in trials:
        weights = sample_weights(train_bins, ratio, mode, recent)
        trial_dir = args.out / mode
        trial_dir.mkdir(parents=True,exist_ok=True)
        print(f'\n=== {mode} / {args.device.upper()} ===', flush=True)
        print(f'[weights] min={weights.min():.3f} max={weights.max():.3f} mean={weights.mean():.3f}', flush=True)
        t0=time.time()
        kwargs = dict(iterations=args.iterations, depth=args.depth,
                      learning_rate=args.learning_rate, l2_leaf_reg=args.l2,
                      loss_function='Logloss', eval_metric='AUC',
                      random_seed=42, thread_count=args.threads,
                      task_type='GPU' if args.device=='gpu' else 'CPU',
                      verbose=100, allow_writing_files=False)
        if args.device=='gpu':
            kwargs.update(devices=str(args.gpu_id), border_count=128)
        model = CatBoostClassifier(**kwargs)
        train_pool = Pool(xt, label=np.asarray(y), weight=weights)
        # Unweighted validation for a consistent early-stop criterion across variants.
        valid_pool = Pool(va, label=np.asarray(yv))
        model.fit(train_pool,eval_set=valid_pool,
                  early_stopping_rounds=args.patience,use_best_model=True)
        p = model.predict_proba(va)[:,1]
        if not np.isfinite(p).all():
            raise SystemExit('STOP: nonfinite predictions')
        metrics = report_metrics(yv, p, ctx, val_w)
        metrics['weighted_target_auc'] = float(roc_auc_score(yv,p,sample_weight=val_w))
        record={'mode':mode,'device':args.device,'iterations':int(model.tree_count_),
                'seconds':round(time.time()-t0,1), **metrics}
        print('[RESULT]',record,flush=True)
        np.save(trial_dir/'val_pred.npy',p.astype(np.float32))
        model.save_model(str(trial_dir/'model.cbm'))
        (trial_dir/'metrics.json').write_text(json.dumps(record,indent=2)+'\n')
        outputs.append(record)
        summary = pd.DataFrame(outputs).sort_values('weighted_target_auc',ascending=False)
        summary.to_csv(args.out/'summary.csv',index=False)
        print('\n=== SUMMARY (target-weighted metric first) ===\n'+summary.to_string(index=False),flush=True)
    print('\nNO SUBMISSION CREATED. These results do not guarantee future-week performance.',flush=True)

if __name__=='__main__': main()
