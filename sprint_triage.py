#!/usr/bin/env python3
"""Offline TechArena CTR triage: compare cached validation predictions fairly.

Run from repository root:
  uv run python sprint_triage.py
No training, no evaluation labels, and no submission generation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc(y, p, w=None):
    if len(np.unique(y)) != 2:
        return float('nan')
    return float(roc_auc_score(y, p, sample_weight=w))


def flatten_events(event_positions):
    positions = np.asarray(event_positions, dtype=np.int64)
    return (positions[:, None] * 10 + np.arange(10)[None, :]).ravel()


def load_old_parquet(path: Path, labels: np.ndarray, n_events: int):
    df = pd.read_parquet(path)
    required = {'event_row', 'label', 'pred'}
    if not required.issubset(df.columns):
        raise ValueError(f'{path}: expected columns {required}; got {df.columns.tolist()}')
    if len(df) != len(labels):
        raise ValueError(f'{path}: {len(df)} predictions != {len(labels)} labels')
    expected_event = np.repeat(np.arange(n_events), 10)
    if not np.array_equal(df['event_row'].to_numpy(), expected_event):
        raise ValueError(f'{path}: event_row not aligned with validation events')
    if not np.array_equal(df['label'].to_numpy(), labels):
        raise ValueError(f'{path}: LABELS DO NOT MATCH cached validation data')
    return np.asarray(df['pred'], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, default=Path('cache/features_gbdt_v5'))
    ap.add_argument('--meta', type=Path, default=Path('cache/gbdt_data.npz'))
    ap.add_argument('--old-parquet', type=Path, default=Path('artifacts/gbdt_position_v5/val_preds_gbdt.parquet'))
    ap.add_argument('--sprint-root', type=Path, default=Path('artifacts/final_sprint_offline/lightgbm'))
    ap.add_argument('--out', type=Path, default=Path('artifacts/sprint_triage_comparison.csv'))
    args = ap.parse_args()

    xy = np.load(args.cache / 'val_matrix.npy', mmap_mode='r')
    y = np.load(args.cache / 'val_matrix_labels.npy', mmap_mode='r')
    if xy.shape[0] != len(y) or xy.shape[1] != 355 or len(y) % 10:
        raise ValueError('Expected validated v5 355-column cache, 10 rows per event')
    if not np.isin(y, [0,1]).all() or np.any(y.reshape(-1,10).sum(axis=1) > 1):
        raise ValueError('Invalid validation labels')
    n_events = len(y) // 10

    with np.load(args.meta) as d:
        ts = np.asarray(d['tr_ts'])
        users = np.asarray(d['tr_user'])
        lbl = np.asarray(d['tr_labels'])
        ev_hist = np.asarray(d['ev_hist'])
    cut = int(np.quantile(ts, 0.85))
    indices = np.flatnonzero(ts >= cut)
    if len(indices) != n_events or not np.array_equal(lbl[indices].ravel(), y):
        raise ValueError('STOP: cached metadata differs from validation labels')
    _, first_idx = np.unique(users[indices], return_index=True)
    first_idx = np.sort(first_idx)
    first_rows = flatten_events(first_idx)
    late_indices = np.flatnonzero(ts[indices] >= np.median(ts[indices]))
    late_rows = flatten_events(late_indices)

    vlens = np.asarray(xy[::10,6], dtype=np.int16)
    elens = (ev_hist >= 0).sum(axis=1)
    bins = [0,3,7,15]
    vb = np.digitize(vlens, bins, right=True)
    eb = np.digitize(elens, bins, right=True)
    vc = np.bincount(vb, minlength=5).astype(float)+1
    ec = np.bincount(eb, minlength=5).astype(float)+1
    ratios = np.clip((ec/ec.sum()) / (vc/vc.sum()), 0.25,4.0)
    weights = np.repeat(ratios[vb], 10)
    ratio_counts = [(int((vb==i).sum()), int((eb==i).sum())) for i in range(5)]
    print('[cohort] validation candidates:', len(y), 'events:', n_events,
          'first-user events:', len(first_idx), 'eval events:', len(elens))
    print('[hist] bins: 0, 1-3, 4-7, 8-15, 16-20')
    print('[hist] (validation count, eval count) per bin:', ratio_counts)
    print('[hist] event reweight ratios:', np.round(ratios,3).tolist())

    preds = {}
    if args.old_parquet.exists():
        preds['previous_gbdt'] = load_old_parquet(args.old_parquet, y, n_events)
    else:
        print(f'[skip] no old predictions found: {args.old_parquet}')
    if args.sprint_root.exists():
        for path in sorted(args.sprint_root.glob('*/val_pred.npy')):
            arr = np.load(path, mmap_mode='r')
            if arr.shape != y.shape:
                print(f'[skip] {path} shape={arr.shape}, expected {y.shape}')
                continue
            preds[path.parent.name] = np.asarray(arr, dtype=np.float64)
    if not preds:
        raise SystemExit('No cached validation predictions found. Check paths.')

    lines = []
    for name, p in preds.items():
        if not np.isfinite(p).all():
            raise ValueError(f'{name}: predictions have inf/NaN')
        line = dict(model=name,
                    full_auc=auc(y,p),
                    first_user_auc=auc(y[first_rows],p[first_rows]),
                    late_half_auc=auc(y[late_rows],p[late_rows]),
                    history_weighted_auc=auc(y,p,weights))
        for i, label in enumerate(('h0','h1_3','h4_7','h8_15','h16_20')):
            es = np.flatnonzero(vb==i)
            idx = flatten_events(es)
            line[f'{label}_auc'] = auc(y[idx],p[idx]) if len(es) else float('nan')
        lines.append(line)
    df = pd.DataFrame(lines).sort_values('first_user_auc', ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print('\n=== VALIDATION COMPARISON ===')
    print(df.to_string(index=False, float_format=lambda z: f'{z:.6f}'))
    print(f'\nSaved: {args.out}')
    print('NO TRAINING. NO SUBMISSION CREATED.')

if __name__ == '__main__':
    main()
