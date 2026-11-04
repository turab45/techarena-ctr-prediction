#!/usr/bin/env python3
"""Fast OFFLINE TechArena CTR experiments from existing features_gbdt_v5 matrices.

No label leakage: trains on train_matrix only; validates on val_matrix only.
No leaderboard upload, final retraining, or evaluation-set predictions.
Designed as a fast screening tool, NOT proof of future-week generalization.

Feature indices verified against turab45/techarena-ctr-prediction/gbdt.py v5,
2026-11-04. Script checks 355-column cache and candidate ordering.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 14 core columns, 12 history/embedding columns, 27 slate-relative transforms;
# 18 relation columns + 12 relatives; 17 context columns + 10 relatives.
# Important: columns 0:25 are NOT all embedding-free.
SAFE_CORE = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 26, 27, 28]
SAFE_CONTEXT = list(range(83, 110))
# Pure cooccurrence count/lift (not latent relations) and slate relatives.
SAFE_RELATIONS = list(range(53, 63)) + list(range(68, 79))
# Context priors are derived from past histories/clicks (not latent models).
CONTEXT_PRIORS = list(range(259, 291))
SAFE93 = sorted(set(SAFE_CORE + SAFE_CONTEXT + SAFE_RELATIONS + CONTEXT_PRIORS))
HYBRID223 = list(range(191)) + CONTEXT_PRIORS  # no learned warm-16 or ALS blocks

VARIANTS = {
    "stable40": (SAFE_CORE + SAFE_CONTEXT, False),
    "stable93": (SAFE93, False),
    "stable93_recent": (SAFE93, True),
    "hybrid223": (HYBRID223, False),
    "hybrid223_recent": (HYBRID223, True),
    "hybrid_without_pref81": (list(range(110)) + CONTEXT_PRIORS, False),
    "hybrid_pref81_only": (SAFE_CORE + list(range(110, 191)) + CONTEXT_PRIORS, False),
    "hybrid_plus_warm_score": (HYBRID223 + list(range(191, 195)), False),
    "hybrid_plus_warm_scalars": (HYBRID223 + list(range(191, 227)), False),
    "hybrid_plus_warm_all": (HYBRID223 + list(range(191, 259)), False),
    "hybrid_plus_als_scalars": (HYBRID223 + list(range(291, 323)), False),
    "hybrid_plus_als_all": (HYBRID223 + list(range(291, 355)), False),
    "hybrid_plus_both_scalars": (HYBRID223 + list(range(191,227)) + list(range(291,323)), False),
    "full355": (list(range(355)), False),
    "full355_recent": (list(range(355)), True),
}


def load_xy(folder: Path, name: str):
    x_path = folder / f"{name}_matrix.npy"
    y_path = folder / f"{name}_matrix_labels.npy"
    if not x_path.exists() or not y_path.exists():
        raise SystemExit(f"Missing {x_path} or {y_path}: build feature cache first.")
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    if x.ndim != 2 or x.shape[1] != 355:
        raise SystemExit(f"Expected v5 cache with 355 features, got {x.shape} for {name}")
    if x.shape[0] != len(y) or len(y) % 10:
        raise SystemExit(f"{name}: shape mismatch or not 10 rows per event")
    if not np.isin(y, [0, 1]).all() or (y.reshape(-1, 10).sum(axis=1) > 1).any():
        raise SystemExit(f"{name}: invalid 0/1 labels or >1 click per event")
    check = np.asarray(x[:min(len(x), 100000), 13]).reshape(-1, 10)
    if not np.array_equal(check, np.broadcast_to(np.arange(10), check.shape)):
        raise SystemExit(f"{name}: slot-order column 13 is not 0..9: STOP, cache layout differs")
    print(f"[{name}] {len(y):,} candidate rows, {len(y)//10:,} events", flush=True)
    return x, y


def contexts(data_path: Path, x_train, y_train, x_val, y_val):
    """Obtain precise train day boundaries and validation cohort; otherwise return None."""
    if not data_path.exists():
        print("[metadata] no gbdt_data.npz; cohort diagnostics will use event order only", flush=True)
        return None
    with np.load(data_path) as d:
        ts = np.asarray(d["tr_ts"])
        users = np.asarray(d["tr_user"])
        tr_labels = np.asarray(d["tr_labels"])
        ev_hist = np.asarray(d["ev_hist"]) if "ev_hist" in d.files else None
        ev_users = np.asarray(d["ev_user"]) if "ev_user" in d.files else None
    start = int(ts.min())
    cut = int(np.quantile(ts, 0.85))
    val_events = np.flatnonzero(ts >= cut)
    if len(val_events) * 10 != len(x_val):
        raise SystemExit("STOP: validation row count does not match cached event cutoff")
    if not np.array_equal(tr_labels[val_events].reshape(-1), y_val):
        raise SystemExit("STOP: validation labels do not align with cached validation events")
    train_days = [3, 4, 5]
    train_counts = []
    for day in train_days:
        lo, hi = start + day * 86400, start + (day + 1) * 86400
        train_counts.append(int(((ts >= lo) & (ts < hi) & (ts < cut)).sum()))
    if sum(train_counts) * 10 != len(x_train):
        raise SystemExit("STOP: train day boundaries do not match cached matrix")
    train_events = np.concatenate([
        np.flatnonzero((ts >= start + day * 86400) &
                        (ts < start + (day + 1) * 86400) & (ts < cut))
        for day in train_days
    ])
    if not np.array_equal(tr_labels[train_events].reshape(-1), y_train):
        raise SystemExit("STOP: training labels do not align with cached training events")
    # For each validation user, use first validation event, matching eval's usual one-event-per-user population.
    _, indices = np.unique(users[val_events], return_index=True)
    first = np.sort(indices)
    late = np.flatnonzero(ts[val_events] >= np.median(ts[val_events]))
    # Additional fully chronological backtest: train on days 3-4, validate on day 5.
    day5_events = train_events[sum(train_counts[:2]):]
    _, d5_idx = np.unique(users[day5_events], return_index=True)
    d5_first = np.sort(d5_idx)
    d5_late = np.flatnonzero(ts[day5_events] >= np.median(ts[day5_events]))
    print(f"[metadata] train event day counts={train_counts} val first-event cohort={len(first):,}", flush=True)
    if ev_users is not None:
        print(f"[metadata] evaluation events={len(ev_users):,}, unique users={len(np.unique(ev_users)):,}", flush=True)
    return {"train_counts": train_counts, "first": first, "late": late,
            "day5_first": d5_first, "day5_late": d5_late,
            "ev_hlen": None if ev_hist is None else (ev_hist >= 0).sum(axis=1)}


def recency_weights(ctx, num_rows: int):
    if ctx is None:
        raise SystemExit("Recency variant needs cache/gbdt_data.npz to identify day boundaries safely")
    counts = ctx["train_counts"]
    time_weights = np.array([0.5, 1.0, 2.0], np.float32)[:len(counts)]
    weights = np.repeat(time_weights,
                        np.asarray(counts, dtype=np.int64) * 10)
    if len(weights) != num_rows:
        raise ValueError("Bad recency weight lengths")
    return weights


def _row_ids(event_positions):
    e = np.asarray(event_positions, dtype=np.int64)
    return (e[:, None] * 10 + np.arange(10)[None, :]).ravel()


def _auc(y, p, sample_weight=None):
    return float(roc_auc_score(y, p, sample_weight=sample_weight))


def history_reweight(x_val, ctx):
    if ctx is None or ctx["ev_hlen"] is None:
        return None
    vlen = np.asarray(x_val[::10, 6], dtype=np.int16)
    elen = np.asarray(ctx["ev_hlen"], dtype=np.int16)
    # [0], [1..3], [4..7], [8..15], [16..20]
    edges = [0, 3, 7, 15]
    vb = np.digitize(vlen, edges, right=True)
    eb = np.digitize(elen, edges, right=True)
    vc = np.bincount(vb, minlength=5).astype(float) + 1
    ec = np.bincount(eb, minlength=5).astype(float) + 1
    ratio = (ec / ec.sum()) / (vc / vc.sum())
    ratio = np.clip(ratio, 0.25, 4.0)
    print("[hist] reweight ratios (diagnostic, not LB guarantee):", ratio.round(3), flush=True)
    return np.repeat(ratio[vb], 10)


def report_metrics(y, pred, ctx, hist_w=None):
    out = {"full_auc": _auc(y, pred)}
    if ctx is not None:
        for name, key in (("first_user_auc", "first"), ("late_half_auc", "late")):
            rows = _row_ids(ctx[key])
            if len(np.unique(y[rows])) == 2:
                out[name] = _auc(y[rows], pred[rows])
    if hist_w is not None:
        out["history_weighted_auc"] = _auc(y, pred, sample_weight=hist_w)
    # Purely diagnostic; never use these alpha values as evidence of generalization.
    prob = np.clip(pred.astype(np.float64), 1e-6, 1 - 1e-6)
    logit = np.log(prob / (1 - prob))
    event_mean = logit.reshape(-1, 10).mean(axis=1).repeat(10)
    for alpha in (0.25, 0.5, 1.0):
        out[f"slate_center_{alpha}_auc"] = _auc(y, logit - alpha * event_mean)
    return out


def train_lgbm(xt, yt, xv, yv, weights, args):
    import lightgbm as lgb
    if args.device != "cpu":
        raise SystemExit("Use --engine catboost --device gpu (LightGBM CUDA/OpenCL support varies by build)")
    train = lgb.Dataset(xt, label=yt, weight=weights, free_raw_data=True)
    valid = lgb.Dataset(xv, label=yv, reference=train, free_raw_data=True)
    params = dict(objective="binary", metric="auc", max_bin=127,
                  bagging_fraction=0.8, bagging_freq=1,
                  force_col_wise=True, num_threads=args.threads,
                  seed=42, verbosity=-1)
    if args.preset == "original":
        params.update(num_leaves=63, learning_rate=0.03,
                      min_data_in_leaf=100, feature_fraction=0.7,
                      lambda_l2=2.0)
    else:
        params.update(num_leaves=31, learning_rate=0.04,
                      min_data_in_leaf=250, feature_fraction=0.85,
                      lambda_l2=10.0)
    model = lgb.train(params, train,
                      num_boost_round=args.iterations,
                      valid_sets=[valid], valid_names=["heldout"],
                      callbacks=[lgb.early_stopping(args.patience, verbose=False),
                                 lgb.log_evaluation(100)])
    p = model.predict(xv, num_iteration=model.best_iteration,
                      num_threads=args.threads)
    return model, p, int(model.best_iteration)


def train_catboost(xt, yt, xv, yv, weights, args):
    from catboost import CatBoostClassifier
    kwargs = dict(iterations=args.iterations, depth=6,
                  learning_rate=0.055, l2_leaf_reg=10.0,
                  loss_function="Logloss", eval_metric="AUC",
                  random_seed=42, thread_count=args.threads,
                  task_type="GPU" if args.device == "gpu" else "CPU",
                  verbose=100, allow_writing_files=False)
    if args.device == "gpu":
        kwargs["devices"] = "0"
    model = CatBoostClassifier(**kwargs)
    model.fit(xt, np.asarray(yt), sample_weight=weights,
              eval_set=(xv, np.asarray(yv)),
              early_stopping_rounds=args.patience, use_best_model=True)
    p = model.predict_proba(xv)[:, 1]
    return model, p, int(model.tree_count_)


def run_variant(name, cfg, x, y, xv, yv, ctx, hist_w, args):
    cols, recent = cfg
    print(f"\n=== {name} ({len(cols)} features, engine={args.engine}, device={args.device}, preset={args.preset}) ===", flush=True)
    t0 = time.time()
    weights = recency_weights(ctx, len(y)) if recent else None
    xt = np.asarray(x[:, cols], dtype=np.float32)
    va = np.asarray(xv[:, cols], dtype=np.float32)
    if not np.isfinite(xt).all() or not np.isfinite(va).all():
        raise SystemExit(f"Nonfinite cache values for {name}; STOP")
    if args.engine == "lightgbm":
        model, p, rounds = train_lgbm(xt, y, va, yv, weights, args)
    else:
        model, p, rounds = train_catboost(xt, y, va, yv, weights, args)
    if not np.isfinite(p).all():
        raise SystemExit(f"Nonfinite prediction for {name}")
    metrics = report_metrics(yv, np.asarray(p), ctx, hist_w)
    dest = args.out / args.engine / name
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "val_pred.npy", np.asarray(p, dtype=np.float32))
    if args.engine == "lightgbm":
        model.save_model(str(dest / "model.txt"))
    else:
        model.save_model(str(dest / "model.cbm"))
    weight_str = ",".join(["0.5", "1", "2"][:len(ctx["train_counts"])]) if recent else "uniform"
    result = dict(variant=name, engine=args.engine, device=args.device,
                  features=len(cols), columns=cols, weights=weight_str,
                  rounds=rounds, seconds=round(time.time() - t0, 1), **metrics)
    (dest / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print("[RESULT]", {k: v for k, v in result.items() if k.endswith("auc")}, flush=True)
    del xt, va, model, p
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("cache/features_gbdt_v5"))
    parser.add_argument("--meta", type=Path, default=Path("cache/gbdt_data.npz"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/final_sprint_offline"))
    parser.add_argument("--variants", default="stable93,hybrid223,hybrid223_recent")
    parser.add_argument("--engine", choices=("lightgbm", "catboost"), default="lightgbm")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--backtest-day5", action="store_true",
                        help="independent chronology: days 3-4 train, day 5 holdout")
    parser.add_argument("--iterations", type=int, default=850)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--preset", choices=("regularized", "original"), default="regularized",
                        help="LightGBM params; original matches gbdt.py (single seed)")
    args = parser.parse_args()
    args.out = args.out / args.preset
    if args.device == "gpu" and args.engine != "catboost":
        parser.error("For GPU use --engine catboost --device gpu")
    if args.threads < 1 or args.iterations < 2 or args.patience < 1:
        parser.error("Invalid threads/iterations/patience")
    requested = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not requested or any(v not in VARIANTS for v in requested):
        parser.error(f"Choose valid variants: {list(VARIANTS)}")
    x, y = load_xy(args.cache, "train")
    xv, yv = load_xy(args.cache, "val")
    ctx = contexts(args.meta, x, y, xv, yv)
    if args.backtest_day5:
        if ctx is None:
            parser.error("--backtest-day5 requires gbdt_data.npz")
        split = sum(ctx["train_counts"][:2]) * 10
        if split <= 0 or split >= len(y):
            parser.error("Day5 backtest has no rows")
        xv, yv = x[split:], y[split:]
        x, y = x[:split], y[:split]
        ctx = {"train_counts": ctx["train_counts"][:2],
               "first": ctx["day5_first"], "late": ctx["day5_late"],
               "ev_hlen": None}
        args.out = args.out / "backtest_day5"
        print(f"[backtest] train={len(y):,} rows on days 3-4; validate={len(yv):,} rows on day 5", flush=True)
    hist_w = history_reweight(xv, ctx)
    all_results = []
    for variant in requested:
        result = run_variant(variant, VARIANTS[variant], x, y, xv, yv, ctx, hist_w, args)
        all_results.append(result)
        args.out.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame([{k:v for k,v in item.items() if k != "columns"}
                                for item in all_results]).sort_values(
            "first_user_auc" if "first_user_auc" in all_results[0] else "full_auc",
            ascending=False)
        summary.to_csv(args.out / f"summary_{args.engine}.csv", index=False)
    print("\n=== OFFLINE SUMMARY ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nNO SUBMISSION CREATED. No guarantee about future-week leaderboard AUC.", flush=True)
    print("Next: verify first-user/late-half robustness and next chronological holdout before any submission.", flush=True)


if __name__ == "__main__":
    main()
