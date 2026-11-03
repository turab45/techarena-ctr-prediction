# gbdt.py
"""
LightGBM pipeline for TechArena CTR prediction.

Independent of pipeline.py (the neural network). Uses cached train.pkl / eval.pkl.
Produces submission_gbdt.csv with rowid,predicted_probability format.

Dependencies:
    uv pip install lightgbm scipy implicit

Usage:
    uv run python gbdt.py                      # full pipeline
    uv run python gbdt.py --skip_als           # quick test without ALS
    uv run python gbdt.py --sample 50000       # subsample events for quick test
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import sparse
from sklearn.metrics import roc_auc_score

try:
    from implicit.als import AlternatingLeastSquares
    HAS_IMPLICIT = True
except ImportError:
    HAS_IMPLICIT = False

CACHE = Path("cache")

FEATURE_COLS = [
    # Item popularity & recency
    'item_ctr', 'item_log_n', 'item_click_count',
    'item_days_since_last_seen', 'item_ctr_last_3d',
    # User activity
    'user_n_events', 'user_log_events', 'user_ctr',
    'user_last_active_gap_days',
    # User-item interaction
    'in_history', 'hist_count', 'hist_last_pos', 'hist_gap_days',
    'als_score',
    # Slate context
    'position', 'n_candidates_in_history',
    'item_ctr_rank_in_slate', 'item_ctr_minus_slate_mean',
    'als_rank_in_slate', 'als_score_minus_slate_mean', 'is_top_als',
    # Temporal
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
    # History summary (event-level, broadcast to all candidates)
    'hist_len', 'hist_unique_ratio',
    'hist_last_item_in_slate', 'hist_overlap_ratio',
    'item2vec_sim_max', 'item2vec_sim_mean', 'item2vec_sim_last',
]


# ---------------------------------------------------------------------------
# Load and split
# ---------------------------------------------------------------------------
def load_events():
    train = pd.read_pickle(CACHE / "train.pkl")
    ev    = pd.read_pickle(CACHE / "eval.pkl")
    print(f"[load] train events={len(train):,}  eval events={len(ev):,}")
    return train, ev


def temporal_split(train, val_frac=0.15):
    cut_ts = int(np.quantile(train['timestamp'].values, 1 - val_frac))
    tr = train[train['timestamp'] <  cut_ts].reset_index(drop=True)
    va = train[train['timestamp'] >= cut_ts].reset_index(drop=True)
    print(f"[split] train={len(tr):,}  val={len(va):,}  "
          f"cutoff={pd.to_datetime(cut_ts, unit='s', utc=True)}")
    return tr, va, cut_ts


# ---------------------------------------------------------------------------
# Explode events to one row per (event, candidate)
# ---------------------------------------------------------------------------
def explode_candidates(df, has_labels=True):
    n = len(df)
    counts = df['items'].apply(len).values
    total = int(counts.sum())

    event_row = np.repeat(np.arange(n), counts)
    position  = np.concatenate([np.arange(c) for c in counts])
    items     = np.concatenate(df['items'].values)

    out = pd.DataFrame({
        'event_row': event_row,
        'position':  position.astype(np.int8),
        'user_id':   df['user_id'].values[event_row],
        'item_id':   items,
        'timestamp': df['timestamp'].values[event_row],
    })
    if has_labels:
        out['label'] = np.concatenate(df['labels'].values).astype(np.int8)
    else:
        out['rowid'] = np.concatenate(df['rowids'].values)

    print(f"[explode] {n:,} events -> {len(out):,} candidates")
    return out


# ---------------------------------------------------------------------------
# Statistics (computed on training portion only — no leakage)
# ---------------------------------------------------------------------------
def compute_item_stats(tr_flat):
    global_ctr = float(tr_flat['label'].mean())
    g = tr_flat.groupby('item_id', sort=False).agg(
        n=('label', 'size'),
        clicks=('label', 'sum'),
        last_ts=('timestamp', 'max'),
    ).reset_index()
    alpha = 5.0
    g['item_ctr']         = ((g['clicks'] + alpha * global_ctr) / (g['n'] + alpha)).astype(np.float32)
    g['item_log_n']       = np.log1p(g['n']).astype(np.float32)
    g['item_click_count'] = g['clicks'].astype(np.float32)
    return g[['item_id', 'last_ts', 'item_ctr', 'item_log_n', 'item_click_count']], global_ctr


def compute_item_ctr_recent(tr_flat, cutoff_ts, window_days=3):
    window_start = cutoff_ts - window_days * 86400
    recent = tr_flat[tr_flat['timestamp'] >= window_start]
    global_ctr = float(tr_flat['label'].mean())
    g = recent.groupby('item_id', sort=False).agg(
        n=('label', 'size'),
        clicks=('label', 'sum'),
    ).reset_index()
    alpha = 5.0
    g['item_ctr_last_3d'] = ((g['clicks'] + alpha * global_ctr) / (g['n'] + alpha)).astype(np.float32)
    return g[['item_id', 'item_ctr_last_3d']]


def compute_user_stats(tr_flat):
    global_ctr = float(tr_flat['label'].mean())
    g = tr_flat.groupby('user_id', sort=False).agg(
        user_n_cand=('label', 'size'),
        user_n_events=('event_row', 'nunique'),
        user_clicks=('label', 'sum'),
        user_last_ts=('timestamp', 'max'),
    ).reset_index()
    alpha = 5.0
    g['user_ctr']        = ((g['user_clicks'] + alpha * global_ctr) / (g['user_n_cand'] + alpha)).astype(np.float32)
    g['user_log_events'] = np.log1p(g['user_n_events']).astype(np.float32)
    return g[['user_id', 'user_n_events', 'user_log_events', 'user_ctr', 'user_last_ts']]


# ---------------------------------------------------------------------------
# ALS collaborative signal
# ---------------------------------------------------------------------------
def train_als(tr_flat, factors=64, iterations=15, reg=0.05):
    if not HAS_IMPLICIT:
        print("[als] implicit not installed — skipping ALS features")
        return None, {}, {}

    clicks = tr_flat[tr_flat['label'] == 1]
    user_ids = sorted(tr_flat['user_id'].unique())
    item_ids = sorted(tr_flat['item_id'].unique())
    user2row = {u: i for i, u in enumerate(user_ids)}
    item2col = {it: i for i, it in enumerate(item_ids)}

    rows = clicks['user_id'].map(user2row).values
    cols = clicks['item_id'].map(item2col).values
    data = np.ones(len(rows), dtype=np.float32)

    R = sparse.csr_matrix((data, (rows, cols)),
                          shape=(len(user_ids), len(item_ids)))
    print(f"[als] matrix shape={R.shape}  nnz={R.nnz:,}")

    t0 = time.time()
    als = AlternatingLeastSquares(
        factors=factors, iterations=iterations,
        regularization=reg, random_state=42, use_gpu=False,
    )
    als.fit(R)
    print(f"[als] trained in {time.time()-t0:.1f}s")

    return (als.user_factors.astype(np.float32),
            als.item_factors.astype(np.float32)), user2row, item2col


def compute_als_scores(df, als_artifacts, user2row, item2col):
    if als_artifacts is None:
        return np.zeros(len(df), dtype=np.float32)
    uf, itf = als_artifacts
    u_idx = df['user_id'].map(user2row).fillna(-1).astype(np.int64).values
    i_idx = df['item_id'].map(item2col).fillna(-1).astype(np.int64).values
    out = np.zeros(len(df), dtype=np.float32)
    valid = (u_idx >= 0) & (i_idx >= 0)
    if valid.any():
        out[valid] = np.einsum('ij,ij->i', uf[u_idx[valid]], itf[i_idx[valid]])
    return out


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------
def add_item2vec_features(flat, events):
    with open(CACHE / "item2vec.pkl", "rb") as f:
        item2vec = pickle.load(f)
    if not item2vec:
        raise ValueError("Item2Vec cache contains no embeddings")
    item2idx = {item: i for i, item in enumerate(item2vec)}
    emb_matrix = np.stack(list(item2vec.values())).astype(np.float32)
    if emb_matrix.ndim != 2 or emb_matrix.shape[1] != 64 or not np.isfinite(emb_matrix).all():
        raise ValueError("Item2Vec embeddings must be finite 64-dimensional vectors")
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix /= np.maximum(norms, 1e-8)

    max_hist = max(events['history_items'].map(len), default=0)
    hist_idx = np.full((len(events), max_hist), -1, dtype=np.int32)
    for event_row, history in enumerate(events['history_items']):
        hist_idx[event_row, :len(history)] = [item2idx.get(item, -1) for item in history]
    cand_idx = flat['item_id'].map(item2idx).fillna(-1).to_numpy(dtype=np.int32)
    event_rows = flat['event_row'].to_numpy(dtype=np.int64)
    result = np.zeros((len(flat), 3), dtype=np.float32)

    # Bound embedding gathers instead of materializing the full candidate-history join.
    if max_hist:
        positions = np.arange(max_hist)[None, :]
        for start in range(0, len(flat), 4096):
            stop = min(start + 4096, len(flat))
            candidates = cand_idx[start:stop]
            histories = hist_idx[event_rows[start:stop]]
            valid = (candidates[:, None] >= 0) & (histories >= 0)
            sims = np.einsum('bd,bhd->bh',
                             emb_matrix[np.maximum(candidates, 0)],
                             emb_matrix[np.maximum(histories, 0)])
            counts = valid.sum(axis=1)
            result[start:stop, 0] = np.where(
                counts > 0, np.where(valid, sims, -np.inf).max(axis=1), 0.0)
            result[start:stop, 1] = np.where(valid, sims, 0.0).sum(axis=1) / np.maximum(counts, 1)
            last_pos = np.where(valid, positions, -1).max(axis=1)
            result[start:stop, 2] = np.where(
                counts > 0, sims[np.arange(stop - start), np.maximum(last_pos, 0)], 0.0)
    flat[['item2vec_sim_max', 'item2vec_sim_mean', 'item2vec_sim_last']] = result
    # Dot(candidate, mean(history)) equals the mean candidate-history similarity.
    flat['item2vec_sim_user_profile'] = result[:, 1]
    emb_cols = [f'item2vec_dim_{i}' for i in range(64)]
    padded_embeddings = np.concatenate([emb_matrix, np.zeros((1, 64), dtype=np.float32)])
    candidate_embeddings = padded_embeddings[np.where(cand_idx >= 0, cand_idx, len(item2idx))]
    flat = pd.concat([flat, pd.DataFrame(candidate_embeddings, columns=emb_cols,
                                        index=flat.index)], axis=1)
    return flat


def add_features(flat, events, item_stats, item_recent, user_stats,
                 als_artifacts, user2row, item2col, global_ctr, cutoff_ts):
    t0 = time.time()

    # --- Item stats ---
    flat = flat.merge(item_stats, on='item_id', how='left')
    flat = flat.merge(item_recent, on='item_id', how='left')
    flat['item_ctr']         = flat['item_ctr'].fillna(global_ctr).astype(np.float32)
    flat['item_log_n']       = flat['item_log_n'].fillna(0).astype(np.float32)
    flat['item_click_count'] = flat['item_click_count'].fillna(0).astype(np.float32)
    flat['item_ctr_last_3d'] = flat['item_ctr_last_3d'].fillna(global_ctr).astype(np.float32)
    gap_days = (flat['timestamp'] - flat['last_ts']).clip(lower=0) / 86400.0
    flat['item_days_since_last_seen'] = np.log1p(gap_days).astype(np.float32).fillna(0.0)
    flat = flat.drop(columns=['last_ts'])

    # --- User stats ---
    flat = flat.merge(user_stats, on='user_id', how='left')
    flat['user_n_events']  = flat['user_n_events'].fillna(0).astype(np.float32)
    flat['user_log_events']= flat['user_log_events'].fillna(0).astype(np.float32)
    flat['user_ctr']       = flat['user_ctr'].fillna(global_ctr).astype(np.float32)
    gap_days = (flat['timestamp'] - flat['user_last_ts']).clip(lower=0) / 86400.0
    flat['user_last_active_gap_days'] = np.log1p(gap_days).astype(np.float32).fillna(0.0)
    flat = flat.drop(columns=['user_last_ts'])

    # --- History interaction ---
    ev_hist = events[['history_items', 'history_timestamps']].copy().reset_index(drop=True)
    ev_hist['event_row'] = ev_hist.index.astype(np.int64)
    ev_hist['hist_pos']   = ev_hist['history_items'].apply(lambda x: list(range(len(x))))
    ev_hist['hist_len']   = ev_hist['history_items'].apply(len).astype(np.float32)
    ev_hist['hist_uniq']  = ev_hist['history_items'].apply(lambda x: len(set(x))).astype(np.float32)

    hist_long = ev_hist[['event_row', 'history_items', 'history_timestamps', 'hist_pos']].explode(
        ['history_items', 'history_timestamps', 'hist_pos']
    ).rename(columns={
        'history_items':      'item_id',
        'history_timestamps': 'hist_ts',
    })
    hist_long = hist_long.dropna(subset=['item_id']).copy()
    hist_long['hist_pos'] = hist_long['hist_pos'].astype(np.int16)

    hist_agg = hist_long.groupby(['event_row', 'item_id'], sort=False).agg(
        hist_count=('hist_pos', 'size'),
        hist_last_ts=('hist_ts', 'max'),
        hist_last_pos=('hist_pos', 'max'),
    ).reset_index()

    flat = flat.merge(hist_agg, on=['event_row', 'item_id'], how='left')
    flat = add_item2vec_features(flat, events)
    flat['in_history']    = flat['hist_count'].notna().astype(np.int8)
    flat['hist_count']    = flat['hist_count'].fillna(0).astype(np.int8)
    flat['hist_last_pos'] = flat['hist_last_pos'].fillna(-1).astype(np.int16)
    gap_vals = flat['timestamp'].astype(float) - flat['hist_last_ts'].astype(float)
    gap_vals = gap_vals.fillna(-1.0).clip(lower=-1.0) / 86400.0
    flat['hist_gap_days'] = gap_vals.astype(np.float32)
    flat = flat.drop(columns=['hist_last_ts'])

    # --- Event-level history summary ---
    ev_summary = ev_hist[['event_row', 'hist_len', 'hist_uniq']].set_index('event_row')
    flat['hist_len']          = flat['event_row'].map(ev_summary['hist_len']).astype(np.float32)
    flat['hist_unique_ratio'] = (
        flat['event_row'].map(ev_summary['hist_uniq']) / flat['hist_len'].clip(lower=1)
    ).astype(np.float32)

    # hist_last_item_in_slate — per-event flag, broadcast to candidates
    last_items = hist_long.sort_values(['event_row', 'hist_pos']).groupby('event_row').tail(1)
    last_items = last_items[['event_row', 'item_id']].rename(columns={'item_id': 'last_hist_item'})
    merged = flat[['event_row', 'item_id']].merge(last_items, on='event_row', how='left')
    row_is_last = (merged['item_id'] == merged['last_hist_item']).astype(np.int8)
    tmp = flat[['event_row']].copy()
    tmp['has_last'] = row_is_last.values
    event_flag = tmp.groupby('event_row')['has_last'].max()
    flat['hist_last_item_in_slate'] = flat['event_row'].map(event_flag).astype(np.int8)

    # hist_overlap_ratio — per-event, distinct history items present among candidates
    hist_set_dict = dict(zip(ev_hist['event_row'], ev_hist['history_items'].apply(set)))
    slate_set_dict = flat.groupby('event_row')['item_id'].apply(set).to_dict()
    overlap_dict = {}
    for eid, h_set in hist_set_dict.items():
        s_set = slate_set_dict.get(eid, set())
        overlap_dict[eid] = len(h_set & s_set) / 10.0 if s_set else 0.0
    flat['hist_overlap_ratio'] = (
        flat['event_row'].map(overlap_dict).fillna(0.0).astype(np.float32)
    )

    # --- ALS ---
    flat['als_score'] = compute_als_scores(flat, als_artifacts, user2row, item2col)

    # --- Slate context ranks ---
    g = flat.groupby('event_row', sort=False)
    flat['item_ctr_rank_in_slate']       = g['item_ctr'].rank(method='average').astype(np.float32)
    flat['item_ctr_minus_slate_mean']    = (flat['item_ctr'] - g['item_ctr'].transform('mean')).astype(np.float32)
    flat['als_rank_in_slate']            = g['als_score'].rank(method='average').astype(np.float32)
    flat['als_score_minus_slate_mean']   = (flat['als_score'] - g['als_score'].transform('mean')).astype(np.float32)
    flat['is_top_als']                   = (flat['als_score'] == g['als_score'].transform('max')).astype(np.int8)
    flat['n_candidates_in_history']      = g['in_history'].transform('sum').astype(np.float32)

    # --- Temporal ---
    dt_arr = pd.to_datetime(flat['timestamp'].values, unit='s', utc=True)
    hour = dt_arr.hour.values + dt_arr.minute.values / 60.0
    flat['hour_sin'] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    flat['hour_cos'] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    dow = dt_arr.weekday.values
    flat['dow_sin']    = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    flat['dow_cos']    = np.cos(2 * np.pi * dow / 7).astype(np.float32)
    flat['is_weekend'] = (dow >= 5).astype(np.int8)

    with open(CACHE / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    flat['user_idx'] = flat['user_id'].map(vocab['user2idx']).fillna(0).astype(np.int32)
    flat['item_idx'] = flat['item_id'].map(vocab['item2idx']).fillna(0).astype(np.int32)

    print(f"[feat] {len(flat):,} rows in {time.time()-t0:.1f}s")
    return flat


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_lgb(tr_flat, va_flat, feature_cols, num_boost_round=5000):
    X_tr = tr_flat[feature_cols].astype(np.float32).values
    y_tr = tr_flat['label'].astype(np.int8).values
    X_va = va_flat[feature_cols].astype(np.float32).values
    y_va = va_flat['label'].astype(np.int8).values

    print(f"[train] X={X_tr.shape}  pos_rate={y_tr.mean():.4f}")

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.03,
        'num_leaves': 255,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l2': 1.0,
        'cat_smooth': 20,
        'cat_l2': 10,
        'verbose': -1,
        'num_threads': 16,
        'seed': 42,
    }

    categorical_cols = [c for c in ('user_idx', 'item_idx') if c in feature_cols]
    dtrain = lgb.Dataset(X_tr, y_tr, feature_name=feature_cols,
                        categorical_feature=categorical_cols)
    dval   = lgb.Dataset(X_va, y_va, reference=dtrain, feature_name=feature_cols)

    t0 = time.time()
    model = lgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval], valid_names=['val'],
        callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(100)],
    )
    print(f"[train] done in {time.time()-t0:.1f}s  best_iter={model.best_iteration}")

    val_preds = model.predict(X_va, num_iteration=model.best_iteration)
    val_auc   = roc_auc_score(y_va, val_preds)
    print(f"\n[RESULT] LightGBM val AUC = {val_auc:.5f}\n")

    imp = pd.DataFrame({
        'feature': feature_cols,
        'gain': model.feature_importance('gain'),
    }).sort_values('gain', ascending=False)
    print("[imp] top 15 features:")
    print(imp.head(15).to_string(index=False))
    return model, val_auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--num_boost_round", type=int, default=5000)
    p.add_argument("--skip_als", action="store_true")
    p.add_argument("--sample", type=int, default=0,
                   help="Subsample N events per split (0=all). Use for quick tests.")
    args = p.parse_args()

    if not HAS_IMPLICIT and not args.skip_als:
        print("[warn] 'implicit' not installed — ALS will be zero.")
        print("[warn] For full signal: uv pip install implicit, then re-run.")

    train_ev, eval_ev = load_events()
    tr_ev, va_ev, cutoff = temporal_split(train_ev, args.val_frac)

    if args.sample:
        tr_ev = tr_ev.sample(min(args.sample, len(tr_ev)), random_state=42).reset_index(drop=True)
        va_ev = va_ev.sample(min(args.sample, len(va_ev)), random_state=42).reset_index(drop=True)
        print(f"[sample] tr={len(tr_ev):,}  va={len(va_ev):,}  (eval unchanged)")

    tr_flat = explode_candidates(tr_ev, has_labels=True)
    va_flat = explode_candidates(va_ev, has_labels=True)
    ev_flat = explode_candidates(eval_ev, has_labels=False)

    print("[stats] computing item stats...")
    item_stats, global_ctr = compute_item_stats(tr_flat)
    print(f"[stats] {len(item_stats):,} items  global_ctr={global_ctr:.5f}")

    print("[stats] computing recent item CTR...")
    item_recent = compute_item_ctr_recent(tr_flat, cutoff)

    print("[stats] computing user stats...")
    user_stats = compute_user_stats(tr_flat)
    print(f"[stats] {len(user_stats):,} users")

    if args.skip_als:
        print("[als] skipped via --skip_als")
        als_artifacts, user2row, item2col = None, {}, {}
    else:
        als_artifacts, user2row, item2col = train_als(tr_flat)

    print("[feat] building train features...")
    tr_flat = add_features(tr_flat, tr_ev, item_stats, item_recent, user_stats,
                            als_artifacts, user2row, item2col, global_ctr, cutoff)
    print("[feat] building val features...")
    va_flat = add_features(va_flat, va_ev, item_stats, item_recent, user_stats,
                            als_artifacts, user2row, item2col, global_ctr, cutoff)
    print("[feat] building eval features...")
    ev_flat = add_features(ev_flat, eval_ev, item_stats, item_recent, user_stats,
                            als_artifacts, user2row, item2col, global_ctr, cutoff)

    # Drop constant features (won't split usefully)
    feature_cols = FEATURE_COLS.copy()
    feature_cols.extend([f'item2vec_dim_{i}' for i in range(64)])
    feature_cols.extend(['user_idx', 'item_idx', 'item2vec_sim_user_profile'])
    constant = [c for c in feature_cols if tr_flat[c].nunique() <= 1]
    if constant:
        print(f"[warn] dropping constant features: {constant}")
        for c in constant:
            feature_cols.remove(c)

    model, val_auc = train_lgb(tr_flat, va_flat, feature_cols, args.num_boost_round)
    model.save_model("gbdt_model.txt")
    print("[save] gbdt_model.txt")

    # --- Predict eval and save submission ---
    X_ev = ev_flat[feature_cols].astype(np.float32).values
    ev_preds = model.predict(X_ev, num_iteration=model.best_iteration)

    sub = pd.DataFrame({
        'rowid': ev_flat['rowid'].values,
        'predicted_probability': ev_preds,
    })
    sub.to_csv("submission_gbdt.csv", index=False)
    print(f"[save] submission_gbdt.csv  rows={len(sub)}")
    print(f"[save] val AUC = {val_auc:.5f}")

    # Save val predictions for later blend analysis
    X_va = va_flat[feature_cols].astype(np.float32).values
    va_out = pd.DataFrame({
        'event_row': va_flat['event_row'].values,
        'item_id':   va_flat['item_id'].values,
        'label':     va_flat['label'].values,
        'pred':      model.predict(X_va, num_iteration=model.best_iteration),
    })
    va_out.to_parquet("val_preds_gbdt.parquet")
    print(f"[save] val_preds_gbdt.parquet  rows={len(va_out):,}")


if __name__ == "__main__":
    main()
