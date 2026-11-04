"""
strict_temporal_v3.py  -  builds on strict_temporal_v2 (same cached matrices, same LightGBM params).

  uv run python3 strict_temporal_v3.py audit                # KS drift per feature: Day5 vs Day6 (natural) and Day6 vs EVAL
  uv run python3 strict_temporal_v3.py peek 5,6,284         # quantiles of given columns in D5 / D6 / eval
  uv run python3 strict_temporal_v3.py sweep                # drop top-K drifting features (default K = 0,1,5,n_flag)
  uv run python3 strict_temporal_v3.py sweep 0,1,10         # ... or your own K list
  uv run python3 strict_temporal_v3.py final 100            # D5+D6 (D6 x2), 3 seeds -> submission_v3_K100.csv (needs that K in the sweep report)
  WINDOW=all uv run python3 strict_temporal_v3.py sweep 0,100   # train on Days 3+4+5 (adds weekday rows); reports/files get an _all tag

Eval block is built at ONE cutoff (c1737936000 = Mon 2025-01-27 00:00 UTC), so eval rows are 0-7 days staler
than the <=1-day-old features the model trained on. KS(D6,eval) >> KS(D5,D6) marks features hit by that shift.
(No adversarial classifier: with per-day feature snapshots it fingerprints the snapshot and hits AUC ~1.0.)
"""
import sys, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

CACHE = Path("cache")
FEAT = CACHE / "features_gbdt_v5"
D5_ROWS = int(os.environ.get("D5_ROWS", 984530))
THREADS = int(os.environ.get("THREADS", 16))
SEEDS = [42, 7, 2024]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
KS_MIN, RATIO_MIN = 0.05, 3.0          # flagged = shifted by >=0.05 AND >=3x the natural day-to-day shift
OUT = Path(".")
WINDOW = os.environ.get("WINDOW", "d5")      # d5 = train on Day 5 only (v2 behaviour); all = Days 3+4+5
SUF = "" if WINDOW == "d5" else "_all"
assert WINDOW in ("d5", "all")

PARAMS = dict(objective="binary", metric="auc", num_leaves=63, learning_rate=0.03, min_data_in_leaf=100,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, max_bin=127,
              num_threads=THREADS, seed=42, verbosity=-1)


def load():
    xa = np.load(FEAT / "train_matrix.npy", mmap_mode="r")
    ya = np.load(FEAT / "train_matrix_labels.npy")
    sl = slice(-D5_ROWS, None) if WINDOW == "d5" else slice(None)
    xt = np.asarray(xa[sl], dtype=np.float32); yt = ya[sl]
    xv = np.asarray(np.load(FEAT / "val_matrix.npy", mmap_mode="r"), dtype=np.float32)
    yv = np.load(FEAT / "val_matrix_labels.npy")
    xe = np.asarray(np.load(FEAT / "block_eval_c1737936000_v5.npy", mmap_mode="r"), dtype=np.float32)
    for n, x in (("D5", xt), ("D6", xv), ("eval", xe)):
        assert len(x) % 10 == 0, f"{n} rows ({len(x)}) not a multiple of 10: event-based steps need whole 10-row events"
    assert xt.shape[1] == xv.shape[1] == xe.shape[1]
    print(f"[data] window={WINDOW}  train {xt.shape}  D6 {xv.shape}  eval {xe.shape}", flush=True)
    return xt, yt, xv, yv, xe


def sample_events(x, n_events, seed):
    rng = np.random.default_rng(seed)
    ne = len(x) // 10
    s = np.sort(rng.choice(ne, min(n_events, ne), replace=False))
    return np.asarray(x[(s[:, None] * 10 + np.arange(10)).ravel()], dtype=np.float32)


def ks(a, b):
    return np.array([ks_2samp(a[:, j], b[:, j], method="asymp").statistic for j in range(a.shape[1])])


def audit(xt, xv, xe):
    nf = xv.shape[1]
    a5, a6, ae = (sample_events(x, 15000, s) for x, s in ((xt, 1), (xv, 2), (xe, 3)))
    ks56, ks6e = ks(a5, a6), ks(a6, ae)
    ratio = ks6e / np.maximum(ks56, 0.01)
    flagged = np.where((ks6e >= KS_MIN) & (ratio >= RATIO_MIN))[0]
    head = [int(j) for j in flagged[np.argsort(-ks6e[flagged])]]
    seen = set(head)
    order = head + [int(j) for j in np.argsort(-ks6e) if int(j) not in seen]
    rank = {f: r for r, f in enumerate(order)}
    rep = pd.DataFrame({"feature": np.arange(nf), "ks_D5_D6": ks56, "ks_D6_eval": ks6e, "ratio": ratio,
                        "flagged": [j in set(head) for j in range(nf)], "drop_rank": [rank[j] for j in range(nf)]})
    rep.sort_values("drop_rank").to_csv(OUT / "drift_report.csv", index=False)
    (OUT / "drop_order.json").write_text(json.dumps({"order": order, "n_flag": len(head)}))
    print(f"[audit] {len(head)} of {nf} features flagged (KS(D6,eval)>={KS_MIN} and >={RATIO_MIN}x KS(D5,D6))")
    print(rep.sort_values("drop_rank").head(30).round(4).to_string(index=False))
    print("[audit] wrote drift_report.csv, drop_order.json")


def peek(cols, xt, xv, xe):
    for c in cols:
        print(f"column {c}")
        for n, x in (("D5", xt), ("D6", xv), ("eval", xe)):
            v = x[::50, c]
            q = np.round(np.quantile(v, [0, .05, .25, .5, .75, .95, 1]), 4)
            print(f"  {n:>4} quantiles(0,5,25,50,75,95,100%) {q}  unique={len(np.unique(v))}")


def keep_cols(nf, order, k):
    gone = set(order[:k])
    return np.array([c for c in range(nf) if c not in gone])


def event_norm(p, alpha):
    q = p.reshape(-1, 10)
    return (q / np.power(q.sum(axis=1, keepdims=True), alpha)).ravel()


def train_es(xt, yt, xv, yv):
    m = lgb.train(PARAMS, lgb.Dataset(xt, label=yt), num_boost_round=1500,
                  valid_sets=[lgb.Dataset(xv, label=yv)], valid_names=["validation"],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, m.best_iteration, m.predict(xv, num_iteration=m.best_iteration)


def sweep(ks_list, xt, yt, xv, yv):
    info = json.loads((OUT / "drop_order.json").read_text())
    order, nf = info["order"], xv.shape[1]
    if ks_list is None:
        ks_list = sorted({0, 1, 5, info["n_flag"]})
    rpath = OUT / f"sweep_report{SUF}.json"
    report = json.loads(rpath.read_text()) if rpath.exists() else {}
    for k in ks_list:
        k = min(k, len(order))
        cols = keep_cols(nf, order, k)
        m, trees, pv = train_es(np.ascontiguousarray(xt[:, cols]), yt, np.ascontiguousarray(xv[:, cols]), yv)
        auc = float(roc_auc_score(yv, pv))
        by_alpha = {a: float(roc_auc_score(yv, event_norm(pv, a))) for a in ALPHAS}
        best_a = max(by_alpha, key=by_alpha.get)
        if by_alpha[best_a] - by_alpha[0.0] < 0.0002:
            best_a = 0.0
        gain = m.feature_importance("gain"); gain = gain / gain.sum()
        top = [(int(cols[i]), round(float(gain[i]), 3)) for i in np.argsort(-gain)[:10]]
        report[str(k)] = dict(k=k, kept=len(cols), d6_auc=auc, trees=int(trees), alpha=best_a,
                              d6_auc_alpha=by_alpha[best_a], dropped=[int(c) for c in order[:k]])
        print(f"[sweep] K={k:>3} kept={len(cols):>3}  D6 AUC={auc:.5f} (trees={trees})  alpha={best_a} -> {by_alpha[best_a]:.5f}")
        print(f"        top gain (col, share): {top}", flush=True)
        rpath.write_text(json.dumps(report, indent=2))
    print(f"[sweep] wrote {rpath.name}")


def final(k, xt, yt, xv, yv, xe):
    order = json.loads((OUT / "drop_order.json").read_text())["order"]
    rep = json.loads((OUT / f"sweep_report{SUF}.json").read_text())[str(k)]
    order = rep.get("dropped", order[:k])          # exact list used in the sweep (old reports fall back to drop_order.json)
    cols = keep_cols(xv.shape[1], order, k)
    xf = np.concatenate([xt[:, cols], xv[:, cols]]); yf = np.concatenate([yt, yv])
    w = np.ones(len(yf), dtype=np.float32); w[len(yt):] = 2.0
    xk = np.ascontiguousarray(xe[:, cols])
    preds = []
    for s in SEEDS:
        m = lgb.train(dict(PARAMS, seed=s), lgb.Dataset(xf, label=yf, weight=w), num_boost_round=rep["trees"])
        preds.append(m.predict(xk))
        print(f"[final] seed {s} done", flush=True)
    pred = event_norm(np.mean(preds, axis=0), rep["alpha"])
    ev_df = pd.read_pickle(CACHE / "eval.pkl")
    rowids = [x for sl in ev_df["rowids"].values for x in sl]
    assert len(rowids) == len(pred) and len(set(rowids)) == len(rowids)
    path = OUT / f"submission_v3{SUF}_K{k}.csv"
    pd.DataFrame({"rowid": rowids, "predicted_probability": pred}).to_csv(path, index=False)
    print(f"[final] wrote {path} (K={k}, trees={rep['trees']}, alpha={rep['alpha']}, seeds={SEEDS})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "audit"
    xt, yt, xv, yv, xe = load()
    if mode == "audit":
        audit(xt, xv, xe)
    elif mode == "peek":
        peek([int(v) for v in sys.argv[2].split(",")], xt, xv, xe)
    elif mode == "sweep":
        sweep([int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else None, xt, yt, xv, yv)
    elif mode == "final":
        final(int(sys.argv[2]), xt, yt, xv, yv, xe)
    else:
        raise SystemExit("usage: audit | peek C1,C2 | sweep [K1,K2,..] | final K")