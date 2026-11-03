# gbdt.py
"""Single-file GBDT pipeline for the TechArena CTR task.

This ports the reference project's strongest tabular recipe into one file:
  * V6 preference features including candidate position (191)
  * warm-16 choice-factor score + decomposition + aligned factor interactions (68)
  * warm clicked-context priors (32)
  * exposure-aware ALS summaries + aligned ALS interactions (64)

The validation rows are always scored with features built from labels strictly
before the validation cutoff.  LightGBM then learns the candidate-level click
probability.  Run `python gbdt.py` for validation plus the final submission.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from numba import config as numba_config, njit, prange, set_num_threads
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

CACHE = Path("cache")
FEATURE_VERSION = 5
N_FEATURES = 355
FEAT = CACHE / f"features_gbdt_v{FEATURE_VERSION}"
FEAT.mkdir(parents=True, exist_ok=True)
DAY = 86400
DATA_VERSION = 4


def log(*args):
    print(*args, flush=True)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def _encode_dataframe(df, umap, imap, labelled):
    n = len(df)
    user = np.fromiter((umap[x] for x in df["user_id"].values), dtype=np.int32, count=n)
    ts = df["timestamp"].values.astype(np.int64)
    tb = df["time_bucket"].values.astype(np.int8)
    dt = df["day_type"].values.astype(np.int8)

    items = np.fromiter(
        (imap[x] for row in df["items"].values for x in row),
        dtype=np.int32,
        count=n * 10,
    ).reshape(n, 10)
    hist = np.full((n, 20), -1, dtype=np.int32)
    hts = np.full((n, 20), -1, dtype=np.int64)
    for i, (ha, ht) in enumerate(zip(df["history_items"].values,
                                    df["history_timestamps"].values)):
        length = min(len(ha), 20)
        if length:
            hist[i, :length] = [imap[x] for x in ha[:length]]
            hts[i, :length] = ht[:length]

    labels = None
    if labelled:
        labels = np.fromiter(
            (int(x) for row in df["labels"].values for x in row),
            dtype=np.int8,
            count=n * 10,
        ).reshape(n, 10)

    out = dict(user=user, ts=ts, tb=tb, dt=dt, items=items, hist=hist, hts=hts)
    if labels is not None:
        out["labels"] = labels
    return out


def prepare_data(rebuild=False):
    if not all((CACHE / name).exists() for name in ("train.pkl", "eval.pkl")):
        from pipeline import load_events
        log("[data] preparing missing event caches from CSV")
        load_events("data/train.csv", CACHE / "train.pkl", True)
        load_events("data/evaluation.csv", CACHE / "eval.pkl", False)
    data_path = CACHE / "gbdt_data.npz"
    meta_path = CACHE / "gbdt_data_meta.json"
    if data_path.exists() and meta_path.exists() and not rebuild:
        meta = json.loads(meta_path.read_text())
        if int(meta.get("version", 0)) == DATA_VERSION:
            log("[data] loading cached integer arrays")
            with np.load(data_path) as z:
                tr = {k[3:]: z[k] for k in z.files if k.startswith("tr_")}
                ev = {k[3:]: z[k] for k in z.files if k.startswith("ev_")}
            nu, ni, start = int(meta["nu"]), int(meta["ni"]), int(meta["start"])
            item_vals = meta.get("items")
            eval_df = pd.read_pickle(CACHE / "eval.pkl")
            rowids = [x for sl in eval_df["rowids"].values for x in sl]
            return tr, ev, nu, ni, start, rowids, item_vals
        log(f"[data] cache version {meta.get('version')} != {DATA_VERSION}; rebuilding")

    log("[data] encoding train/eval events")
    train = pd.read_pickle(CACHE / "train.pkl")
    evdf = pd.read_pickle(CACHE / "eval.pkl")

    user_vals = pd.unique(pd.concat([train["user_id"], evdf["user_id"]],
                                    ignore_index=True))
    umap = {v: i for i, v in enumerate(user_vals)}

    item_set = set()
    for df in (train, evdf):
        for row in df["items"].values:
            item_set.update(row)
        for row in df["history_items"].values:
            item_set.update(row)
    item_vals = pd.Index(sorted(item_set))
    imap = {v: i for i, v in enumerate(item_vals)}

    tr = _encode_dataframe(train, umap, imap, True)
    ev = _encode_dataframe(evdf, umap, imap, False)
    nu, ni = len(user_vals), len(item_vals)
    start = int(tr["ts"].min())

    # The reference pipeline processes events in CSV/timestamp order.  The pickle
    # cache is sorted by event id, so restore chronological order for every
    # "first event per user" statistic, warm-model fit and block assembly.
    order = np.argsort(tr["ts"], kind="stable")
    if not np.array_equal(order, np.arange(len(order))):
        tr = {k: v[order] for k, v in tr.items()}
    rowids = [x for sl in evdf["rowids"].values for x in sl]

    np.savez(
        data_path,
        **{f"tr_{k}": v for k, v in tr.items()},
        **{f"ev_{k}": v for k, v in ev.items()},
    )
    meta_path.write_text(json.dumps(dict(nu=nu, ni=ni, start=start,
                                         version=DATA_VERSION,
                                         items=list(item_vals))))
    log(f"[data] users={nu:,} items={ni:,} train events={len(tr['ts']):,} "
        f"eval events={len(ev['ts']):,}")
    return tr, ev, nu, ni, start, rowids, list(item_vals)


# ---------------------------------------------------------------------------
# Statistics / factor models (all cut off strictly before `cutoff`)
# ---------------------------------------------------------------------------
def _clicked_events(ds, idx):
    labels = ds["labels"][idx]
    hit = labels.sum(axis=1) == 1
    events = idx[hit]
    items = ds["items"][events, labels[hit].argmax(axis=1)]
    return events, items


def _core_statistics(ds, nu, ni, idx):
    y = ds["labels"][idx].ravel()
    g = float(y.mean())
    ue = np.repeat(ds["user"][idx], 10)
    ie = ds["items"][idx].ravel()
    ui = np.bincount(ue, minlength=nu).astype(np.float32)
    uc = np.bincount(ue, weights=y, minlength=nu).astype(np.float32)
    ii = np.bincount(ie, minlength=ni).astype(np.float32)
    ic = np.bincount(ie, weights=y, minlength=ni).astype(np.float32)

    _, first = np.unique(ds["user"][idx], return_index=True)
    first_events = idx[first]
    h = ds["hist"][first_events]
    hu = np.repeat(ds["user"][first_events], 20)
    hi = h.ravel()
    valid = hi >= 0
    ce, ci = _clicked_events(ds, idx)

    matrix = coo_matrix(
        (np.ones(int(valid.sum()) + len(ci), dtype=np.float32),
         (np.concatenate([hu[valid], ds["user"][ce]]),
          np.concatenate([hi[valid], ci]))),
        shape=(nu, ni),
    ).tocsr()
    matrix.data = np.log1p(matrix.data)
    svd = TruncatedSVD(n_components=32, n_iter=4, random_state=42)
    svd.fit(matrix)
    emb = svd.components_.T.copy()
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    return g, ui, uc, ii, ic, emb.astype(np.float32)


def _personalized_statistics(ds, nu, ni, idx, core):
    _, first = np.unique(ds["user"][idx], return_index=True)
    first_events = idx[first]
    h = ds["hist"][first_events]
    hu = np.repeat(ds["user"][first_events], 20)
    hi = h.ravel()
    valid = hi >= 0
    ce, ci = _clicked_events(ds, idx)

    rows = np.concatenate([hu[valid], ds["user"][ce]])
    cols = np.concatenate([hi[valid], ci])
    matrix = coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(nu, ni)
    ).tocsr()
    matrix.data = np.log1p(matrix.data)
    idf = np.log1p(nu / (1 + np.asarray((matrix > 0).sum(axis=0)).ravel()))
    weighted = normalize(matrix.multiply(idf).tocsr(), norm="l2")
    svd = TruncatedSVD(n_components=64, n_iter=5, random_state=43)
    svd.fit(weighted)
    emb = normalize(svd.components_.T.copy()).astype(np.float32)
    profiles = normalize(matrix @ emb).astype(np.float32)
    return emb, profiles


def _build_warm_statistics(ds, nu, ni, idx, cutoff, core, emb64, profiles):
    ce, ci = _clicked_events(ds, idx)

    # history -> clicked item relation table
    h = ds["hist"][ce]
    left = h.ravel()
    right = np.repeat(ci, 20)
    mask = left >= 0
    matrix = coo_matrix(
        (np.ones(int(mask.sum()), dtype=np.float32),
         (left[mask], right[mask])),
        shape=(ni, ni),
    ).tocsr()

    # warm-up transitions from each user's first available pre-cutoff history
    _, first = np.unique(ds["user"][idx], return_index=True)
    histories = ds["hist"][idx[first]]
    for j in range(1, 20):
        previous = histories[:, :j]
        following = histories[:, j]
        source = previous.ravel()
        target = np.repeat(following, j)
        m = (source >= 0) & (target >= 0)
        matrix += coo_matrix(
            (np.ones(int(m.sum()), dtype=np.float32),
             (source[m], target[m])),
            shape=(ni, ni),
        ).tocsr()

    hc = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    cc = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)
    co = matrix.tocoo()
    keys = co.row.astype(np.int64) * ni + co.col
    order = np.argsort(keys)
    keys = keys[order]
    counts = co.data[order].astype(np.float32)
    co.data /= np.sqrt((hc[co.row] + 10.0) * (cc[co.col] + 10.0))
    svd = TruncatedSVD(n_components=64, n_iter=5, random_state=44)
    us = svd.fit_transform(co.tocsr())
    s = np.maximum(svd.singular_values_, 1e-8)
    src = (us / np.sqrt(s)).astype(np.float32)
    dst = (svd.components_.T * np.sqrt(s)).astype(np.float32)

    # contextual exposure/click summaries
    y = ds["labels"][idx].ravel()
    g = float(y.mean())
    bucket = np.repeat(ds["tb"][idx], 10)
    bucket_impr = np.bincount(bucket, minlength=4)
    bucket_click = np.bincount(bucket, weights=y, minlength=4)
    context = {
        "global": g,
        "bucket_ctr": ((bucket_click + 100.0 * g) / (bucket_impr + 100.0)).astype(np.float32),
    }
    users_flat = np.repeat(ds["user"][idx], 10)
    items_flat = ds["items"][idx].ravel()
    recent_rows = np.repeat(ds["ts"][idx], 10) >= (cutoff - DAY)
    for name, ids, size in (("user", users_flat, nu), ("item", items_flat, ni)):
        key = ids * 4 + bucket
        context[name + "_impr"] = np.bincount(ids, minlength=size)
        context[name + "_click"] = np.bincount(ids, weights=y, minlength=size)
        context[name + "_bucket_impr"] = np.bincount(key, minlength=size * 4)
        context[name + "_bucket_click"] = np.bincount(key, weights=y, minlength=size * 4)
        context[name + "_recent_impr"] = np.bincount(ids[recent_rows], minlength=size)
        context[name + "_recent_click"] = np.bincount(
            ids[recent_rows], weights=y[recent_rows], minlength=size
        )

    relations = ((core, emb64, profiles), keys, counts, hc, cc, src, dst)
    return relations, context


def _preference_statistics(ds, nu, ni, idx, cutoff):
    core = _core_statistics(ds, nu, ni, idx)
    emb64, profiles = _personalized_statistics(ds, nu, ni, idx, core)
    relations, context = _build_warm_statistics(
        ds, nu, ni, idx, cutoff, core, emb64, profiles
    )

    _, keys, counts, hc, cc, _, _ = relations
    row = keys // ni
    col = keys % ni
    matrix = coo_matrix(
        (counts / np.sqrt((hc[row] + 10.0) * (cc[col] + 10.0)), (row, col)),
        shape=(ni, ni),
    ).tocsr()
    svd = TruncatedSVD(n_components=128, n_iter=5, random_state=46)
    us = svd.fit_transform(matrix)
    s = np.maximum(svd.singular_values_, 1e-8)
    left = (us / np.sqrt(s)).astype(np.float32)
    right = (svd.components_.T * np.sqrt(s)).astype(np.float32)
    taste = normalize(right[:, 1:]).astype(np.float32)

    _, first = np.unique(ds["user"][idx], return_index=True)
    first_events = idx[first]
    h = ds["hist"][first_events]
    hu = np.repeat(ds["user"][first_events], 20)
    hi = h.ravel()
    valid = hi >= 0
    ce, ci = _clicked_events(ds, idx)

    users = np.concatenate([hu[valid], ds["user"][ce]])
    items = np.concatenate([hi[valid], ci])
    positive = coo_matrix(
        (np.ones(len(users), dtype=np.float32), (users, items)), shape=(nu, ni)
    ).tocsr()
    exposure = coo_matrix(
        (np.ones(len(idx) * 10, dtype=np.float32),
         (np.repeat(ds["user"][idx], 10), ds["items"][idx].ravel())),
        shape=(nu, ni),
    ).tocsr()
    pc = np.asarray(positive.sum(axis=1)).ravel()
    ec = np.asarray(exposure.sum(axis=1)).ravel()
    pm = (positive @ taste) / np.maximum(pc[:, None], 1)
    em = (exposure @ taste) / np.maximum(ec[:, None], 1)
    profile = normalize(pm).astype(np.float32)
    contrast = normalize(pm - em).astype(np.float32)
    variance = np.maximum(np.mean(taste * taste, axis=0), 1e-6)
    adjusted = normalize((pm - em) / np.sqrt(variance)).astype(np.float32)
    return ((relations, context), left, right, taste, profile,
            contrast, adjusted, pc.astype(np.float32))


def get_preference(ds, nu, ni, cutoff):
    path = FEAT / f"pref_{cutoff}.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    log(f"[stats] preference features cutoff={cutoff} "
        f"({pd.to_datetime(cutoff, unit='s')})")
    idx = np.flatnonzero(ds["ts"] < cutoff)
    t = time.time()
    result = _preference_statistics(ds, nu, ni, idx, cutoff)
    with path.open("wb") as f:
        pickle.dump(result, f, protocol=5)
    log(f"[stats] done in {time.time() - t:.1f}s")
    return result


# ---------------------------------------------------------------------------
# Warm-16 joint choice model
# ---------------------------------------------------------------------------
@njit
def _warm_epoch(order, users, positive, negatives, U, V, au, av,
                lr, reg, weight, pop_bias):
    d = U.shape[1]
    loss = 0.0
    for q in order:
        u = users[q]
        items = negatives[q].copy()
        items[0] = positive[q]
        z = U[u].copy()
        logits = np.empty(len(items))
        maximum = -1e30
        for j in range(len(items)):
            v = pop_bias[items[j]]
            for k in range(d):
                v += z[k] * V[items[j], k]
            logits[j] = max(-25.0, min(25.0, v))
            maximum = max(maximum, logits[j])
        p = np.exp(logits - maximum)
        denom = p.sum()
        loss += math.log(denom) + maximum - logits[0]
        grad = np.zeros(d, np.float32)
        for j in range(len(items)):
            i = items[j]
            err = weight * (p[j] / denom - (1.0 if j == 0 else 0.0))
            for k in range(d):
                grad[k] += err * V[i, k]
                g = err * z[k] + weight * reg * V[i, k]
                av[i, k] += g * g
                V[i, k] -= lr * g / math.sqrt(av[i, k])
        for k in range(d):
            g = grad[k] + weight * reg * U[u, k]
            au[u, k] += g * g
            U[u, k] -= lr * g / math.sqrt(au[u, k])
    return loss / max(len(order), 1)


@njit
def _choice_epoch(order, users, items, history, target,
                  U, V, H, bu, bi, au, av, ah, abu, abi, lr, reg):
    d = U.shape[1]
    total = 0.0
    for e in order:
        u = users[e]
        z = U[u].copy()
        n = 0
        for q in range(20):
            if history[e, q] >= 0:
                n += 1
        scale = 1.0 / math.sqrt(max(n, 1))
        for q in range(n):
            h = history[e, q]
            for k in range(d):
                z[k] += scale * H[h, k]

        logits = np.empty(10)
        maximum = 0.0
        for j in range(10):
            item = items[e, j]
            value = -1.8 + bu[u] + bi[item]
            for k in range(d):
                value += z[k] * V[item, k]
            logits[j] = max(-30.0, min(30.0, value))
            maximum = max(maximum, logits[j])

        p = np.empty(10)
        denom = math.exp(-maximum)
        for j in range(10):
            p[j] = math.exp(logits[j] - maximum)
            denom += p[j]
        total += math.log(denom) + maximum - (
            logits[target[e]] if target[e] < 10 else 0.0
        )

        gradient = np.zeros(d, dtype=np.float32)
        biasgrad = 0.0
        for j in range(10):
            item = items[e, j]
            error = p[j] / denom - (1.0 if target[e] == j else 0.0)
            biasgrad += error
            for k in range(d):
                gradient[k] += error * V[item, k]
                g = error * z[k] + reg * V[item, k]
                av[item, k] += g * g
                V[item, k] -= lr * g / math.sqrt(av[item, k])
            g = error + reg * bi[item]
            abi[item] += g * g
            bi[item] -= lr * g / math.sqrt(abi[item])
        for k in range(d):
            g = gradient[k] + reg * U[u, k]
            au[u, k] += g * g
            U[u, k] -= lr * g / math.sqrt(au[u, k])
        g = biasgrad + reg * bu[u]
        abu[u] += g * g
        bu[u] -= lr * g / math.sqrt(abu[u])

        for q in range(n):
            h = history[e, q]
            for k in range(d):
                g = scale * gradient[k] + reg * H[h, k]
                ah[h, k] += g * g
                H[h, k] -= lr * g / math.sqrt(ah[h, k])
    return total / len(order)


@njit
def _predict_events(users, items, history, U, V, H, bu, bi):
    out = np.empty((len(users), 10), dtype=np.float64)
    d = U.shape[1]
    for e in range(len(users)):
        u = users[e]
        z = U[u].copy()
        n = 0
        for q in range(20):
            if history[e, q] >= 0:
                n += 1
        scale = 1.0 / math.sqrt(max(n, 1))
        for q in range(n):
            h = history[e, q]
            for k in range(d):
                z[k] += scale * H[h, k]
        logits = np.empty(10)
        maximum = 0.0
        for j in range(10):
            item = items[e, j]
            value = -1.8 + bu[u] + bi[item]
            for k in range(d):
                value += z[k] * V[item, k]
            logits[j] = max(-30.0, min(30.0, value))
            maximum = max(maximum, logits[j])
        denom = math.exp(-maximum)
        for j in range(10):
            out[e, j] = math.exp(logits[j] - maximum)
            denom += out[e, j]
        for j in range(10):
            out[e, j] /= denom
    return out


def _warm_history(idx, ds):
    _, first = np.unique(ds["user"][idx], return_index=True)
    events = idx[first]
    h = ds["hist"][events]
    users = np.repeat(ds["user"][events], 20)
    items = h.ravel()
    keep = items >= 0
    stamps = ds["hts"][events].ravel()
    keys = np.rec.fromarrays(
        [users[keep], items[keep], stamps[keep]],
        names="user,item,timestamp",
    )
    _, unique = np.unique(keys, return_index=True)
    return users[keep][unique].astype(np.int32), items[keep][unique].astype(np.int32)


def _event_targets(ds, idx):
    y = ds["labels"][idx]
    return np.where(y.sum(axis=1) > 0, y.argmax(axis=1), 10).astype(np.int32)


def _init_warm_params(ds, nu, ni, init_cut, dim=16):
    s = get_preference(ds, nu, ni, init_cut)
    context = s[0][1]
    V = normalize(s[3][:, :dim]).astype(np.float32)
    U = (0.25 * normalize(s[4][:, :dim])).astype(np.float32)
    H = (0.25 * V.copy()).astype(np.float32)
    g = context["global"]
    ur = (context["user_click"] + 50.0 * g) / (context["user_impr"] + 50.0)
    ir = (context["item_click"] + 50.0 * g) / (context["item_impr"] + 50.0)
    t = np.clip(ur * 10.0, 0.02, 0.98)
    bu = np.clip(np.log(t / (1.0 - t) / 10.0) + 1.8, -2, 2).astype(np.float32)
    bi = np.clip(np.log(ir / g), -2, 2).astype(np.float32)
    return [U, V, H, bu, bi]


def get_warm_model(ds, nu, ni, start, cutoff, dim=16, epochs=10):
    path = FEAT / f"warm16_{cutoff}.npz"
    if path.exists():
        with np.load(path) as z:
            return [z[k] for k in ("U", "V", "H", "bu", "bi")]
    day = (cutoff - start) // DAY
    init_cut = start + max(int(day) - 2, 2) * DAY
    log(f"[warm] fitting cutoff={cutoff} init={init_cut} dim={dim} epochs={epochs}")
    t = time.time()
    idx = np.flatnonzero(ds["ts"] < cutoff)
    params = _init_warm_params(ds, nu, ni, init_cut, dim)
    acc = [np.full_like(p, 0.1) for p in params]

    wu, wi = _warm_history(idx, ds)
    counts = np.bincount(wi, minlength=ni) + 1
    pop_bias = np.log(counts / counts.mean()).astype(np.float32)
    users = ds["user"][idx]
    items = ds["items"][idx]
    history = ds["hist"][idx]
    target = _event_targets(ds, idx)
    rng = np.random.default_rng(151)

    for epoch in range(1, epochs + 1):
        negatives = rng.integers(0, ni, size=(len(wu), 11), dtype=np.int32)
        collision = negatives == wi[:, None]
        while collision.any():
            negatives[collision] = rng.integers(
                0, ni, size=int(collision.sum()), dtype=np.int32
            )
            collision = negatives == wi[:, None]
        _warm_epoch(rng.permutation(len(wu)), wu, wi, negatives,
                    params[0], params[1], acc[0], acc[1],
                    0.045, 0.002, 1.0, pop_bias)
        loss = _choice_epoch(rng.permutation(len(users)), users, items, history,
                             target, *params, *acc, 0.045, 0.002)
        if epoch in (1, 5, 10):
            log(f"[warm] epoch={epoch} loss={loss:.5f} elapsed={time.time() - t:.0f}s")
    np.savez_compressed(path, U=params[0], V=params[1], H=params[2],
                        bu=params[3], bi=params[4])
    log(f"[warm] done in {time.time() - t:.0f}s")
    return params


# ---------------------------------------------------------------------------
# ALS exposure model
# ---------------------------------------------------------------------------
@njit(parallel=True)
def _solve_rows(indptr, indices, weights, targets, factors, other_bias,
                gram, rhs, background, global_rate, prior_bias,
                latent_reg=1.0, bias_reg=20.0):
    n = len(indptr) - 1
    d = factors.shape[1]
    bias = np.empty(n, np.float64)
    latent = np.empty((n, d), np.float64)
    for row in prange(n):
        a = background[row] * gram.copy()
        b = background[row] * rhs.copy()
        a[0, 0] += bias_reg
        b[0] += bias_reg * prior_bias[row]
        for j in range(1, d + 1):
            a[j, j] += latent_reg
        vector = np.empty(d + 1, np.float64)
        vector[0] = 1.0
        for q in range(indptr[row], indptr[row + 1]):
            item = indices[q]
            weight = weights[q]
            for j in range(d):
                vector[j + 1] = factors[item, j]
            residual = targets[q] - weight * (global_rate + other_bias[item])
            for j in range(d + 1):
                b[j] += residual * vector[j]
                for k in range(d + 1):
                    a[j, k] += weight * vector[j] * vector[k]
        value = np.linalg.solve(a, b)
        bias[row] = value[0]
        latent[row] = value[1:]
    return bias, latent


def _augmented(factors):
    return np.column_stack([np.ones(len(factors)), factors])


def _paired_matrices(users, items, weights, positives, shape):
    w = coo_matrix((weights, (users, items)), shape=shape).tocsr()
    t = coo_matrix((positives, (users, items)), shape=shape).tocsr()
    assert np.array_equal(w.indptr, t.indptr)
    assert np.array_equal(w.indices, t.indices)
    return w, t


def _als_prepare(ds, nu, ni, cutoff):
    idx = np.flatnonzero(ds["ts"] < cutoff)
    wu, wi = _warm_history(idx, ds)
    g = float(ds["labels"][idx].ravel().mean())
    ue = np.bincount(np.repeat(ds["user"][idx], 10), minlength=nu)
    ie = np.bincount(ds["items"][idx].ravel(), minlength=ni)
    uc = np.bincount(np.repeat(ds["user"][idx], 10),
                     weights=ds["labels"][idx].ravel(), minlength=nu)
    ic = np.bincount(ds["items"][idx].ravel(),
                     weights=ds["labels"][idx].ravel(), minlength=ni)
    up = (uc + 50.0 * g) / (ue + 50.0) - g
    ip = (ic + 50.0 * g) / (ie + 50.0) - g
    warm_count = np.bincount(wu, minlength=nu)
    confidence = 9.0 * warm_count / ni

    users = np.concatenate([np.repeat(ds["user"][idx], 10), wu])
    items = np.concatenate([ds["items"][idx].ravel(), wi])
    weights = np.ones(len(users), dtype=np.float32)
    positives = np.concatenate([
        ds["labels"][idx].ravel().astype(np.float32),
        np.ones(len(wu), dtype=np.float32),
    ])
    w, t = _paired_matrices(users, items, weights, positives, (nu, ni))
    wt, tt = w.T.tocsr(), t.T.tocsr()
    assert np.array_equal(wt.indptr, tt.indptr)
    assert np.array_equal(wt.indices, tt.indices)
    return w, t, wt, tt, g, up, ip, confidence, ue.astype(np.float64), ie


def get_als_model(ds, nu, ni, start, cutoff):
    path = FEAT / f"als_{cutoff}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    log(f"[als] fitting cutoff={cutoff} ({pd.to_datetime(cutoff, unit='s')})")
    t = time.time()
    w, tr, wt, tt, g, up, ip, confidence, exposures, item_exposures = _als_prepare(
        ds, nu, ni, cutoff
    )
    warm = get_warm_model(ds, nu, ni, start, cutoff)
    V = warm[1].astype(np.float64)
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    U = np.zeros((nu, 16))
    bu = up.copy()
    bi = ip.copy()
    for iteration in range(6):
        x = _augmented(V)
        bu, U = _solve_rows(
            w.indptr, w.indices, w.data, tr.data, V, bi,
            x.T @ x, -x.T @ (g + bi), confidence, g, up,
        )
        x = _augmented(U)
        bi, V = _solve_rows(
            wt.indptr, wt.indices, wt.data, tt.data, U, bu,
            x.T @ (confidence[:, None] * x),
            -x.T @ (confidence * (g + bu)),
            np.ones(ni), g, ip,
        )
    np.savez_compressed(
        path, U=U, V=V, bu=bu, bi=bi, g=g, exposures=exposures,
        item_exposures=item_exposures, cutoff=cutoff, iteration=6,
    )
    log(f"[als] done in {time.time() - t:.1f}s")
    return dict(U=U, V=V, bu=bu, bi=bi, g=g, exposures=exposures,
                item_exposures=item_exposures)


def _als_score(ds, events, model, nu, ni):
    """Return user factor and user bias for each event, folding in cold users."""
    U = model["U"]
    V = model["V"]
    bi = model["bi"]
    g = float(model["g"])
    exposures = model["exposures"]
    users = ds["user"][events]
    u = U[users].copy()
    bias = model["bu"][users].copy()
    cold = exposures[users] == 0
    if cold.any():
        selected = events[cold]
        h = ds["hist"][selected]
        stamps = ds["hts"][selected]
        valid = h >= 0
        local = np.repeat(np.arange(len(h)), 20)[valid.ravel()]
        target_items = h.ravel()[valid.ravel()]
        target_stamps = stamps.ravel()[valid.ravel()]
        keys = np.rec.fromarrays([local, target_items, target_stamps],
                                names="row,item,timestamp")
        _, keep = np.unique(keys, return_index=True)
        local = local[keep]
        target_items = target_items[keep]
        values = np.ones(len(local), dtype=np.float32)
        w, t = _paired_matrices(local, target_items, values, values,
                                (len(h), ni))
        x = _augmented(V)
        count = np.bincount(local, minlength=len(h))
        cb, cu = _solve_rows(
            w.indptr, w.indices, w.data, t.data, V, bi,
            x.T @ x, -x.T @ (g + bi), 9.0 * count / ni, g,
            np.zeros(len(h)),
        )
        u[cold] = cu
        bias[cold] = cb
    return u, bias


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
def _feature_core(ds, events, extra_stats):
    """14 core (including slot position) + 12 extra + 27 relative features."""
    g, ui, uc, ii, ic, emb = extra_stats[0]
    emb64, profiles = extra_stats[1], extra_stats[2]
    n = len(events)
    items = ds["items"][events]
    users = ds["user"][events]
    h = ds["hist"][events]
    valid = h >= 0
    safe = np.maximum(h, 0)
    length = valid.sum(axis=1)

    v = emb[safe] * valid[:, :, None]
    avg = v.sum(axis=1) / np.maximum(length[:, None], 1)
    sim = np.einsum("nhd,ncd->nch", v, emb[items])
    positions = np.arange(1, 21)[None, :]

    cols = [
        ((uc[users] + 50.0 * g) / (ui[users] + 50.0))[:, None].repeat(10, 1),
        (ic[items] + 50.0 * g) / (ii[items] + 50.0),
        np.log1p(ui[users])[:, None].repeat(10, 1),
        np.log1p(ii[items]),
        np.broadcast_to(ds["tb"][events][:, None], (n, 10)),
        np.broadcast_to(ds["dt"][events][:, None], (n, 10)),
        np.broadcast_to(length[:, None], (n, 10)),
        (h[:, None, :] == items[:, :, None]).sum(axis=2),
        (avg[:, None, :] * emb[items]).sum(axis=2),
        sim.max(axis=2),
        (sim * positions).sum(axis=2) / np.maximum(
            (valid * positions).sum(axis=1), 1
        )[:, None],
        ((((ic[safe] + 50.0 * g) / (ii[safe] + 50.0)) * valid).sum(axis=1)
         / np.maximum(length, 1))[:, None].repeat(10, 1),
        np.broadcast_to(((ds["ts"][events] % 86400) / 86400.0)[:, None], (n, 10)),
        np.broadcast_to(np.arange(10), (n, 10)),
    ]
    x = np.stack(cols, axis=2)

    age = length[:, None] - 1 - np.arange(20)[None, :]
    vals = []
    for width in (1, 3, 5, 10):
        mask = valid[:, None, :] & (age[:, None, :] < width)
        vals.append((sim * mask).sum(axis=2) / np.maximum(mask.sum(axis=2), 1))
    for decay in (0.5, 0.85):
        w = np.power(decay, np.maximum(age, 0)) * valid
        vals.append((sim * w[:, None, :]).sum(axis=2)
                    / np.maximum(w.sum(axis=1), 1e-9)[:, None])
    vals.extend([
        sim.sum(axis=2) / np.maximum(length, 1)[:, None],
        sim.max(axis=2),
        np.sqrt((sim ** 2).sum(axis=2) / np.maximum(length, 1)[:, None]),
        (profiles[users][:, None, :] * emb64[items]).sum(axis=2),
        ((h[:, None, :] == items[:, :, None])
         & (age[:, None, :] < 3)).sum(axis=2),
        np.where(valid[:, None, :], sim, -2).argmax(axis=2)
        / np.maximum(length, 1)[:, None],
    ])
    x = np.concatenate([x, np.stack(vals, axis=2)], axis=2)

    rel = []
    # Position was inserted at index 13. Preserve every existing similarity
    # feature's relative transforms rather than accidentally shifting them.
    # Relative transforms of position itself are redundant with position.
    for col in (1, 8, 9, 14, 15, 16, 20, 21, 23):
        vv = x[:, :, col]
        rel.extend([
            vv - vv.mean(axis=1, keepdims=True),
            vv - vv.max(axis=1, keepdims=True),
            np.argsort(np.argsort(vv, axis=1), axis=1) / 9.0,
        ])
    x = np.concatenate([x] + [r[:, :, None] for r in rel], axis=2)
    return x.reshape(-1, x.shape[2])


def _feature_relations(ds, events, relations):
    extra_stats, keys, counts, hc, cc, src, dst = relations
    n = len(events)
    items = ds["items"][events]
    h = ds["hist"][events]
    valid = h >= 0
    safe = np.maximum(h, 0)
    length = valid.sum(axis=1)
    x = _feature_core(ds, events, extra_stats)

    query = safe[:, None, :].astype(np.int64) * (cc.shape[0]) + items[:, :, None]
    pos = np.searchsorted(keys, query)
    inside = pos < len(keys)
    pos = np.minimum(pos, len(keys) - 1)
    c = np.where(
        inside & (keys[pos] == query) & valid[:, None, :], counts[pos], 0.0
    ).astype(np.float32)
    hc_h = hc[safe][:, None, :]
    cc_i = cc[items][:, :, None]
    cosine = c / np.sqrt((hc_h + 10.0) * (cc_i + 10.0))
    lift = c / (c + 3.0) * np.log1p(
        c * counts.sum() / ((hc_h + 10.0) * (cc_i + 10.0))
    )
    latent = np.einsum("nhd,ncd->nch", src[safe], dst[items]) * valid[:, None, :]

    age = length[:, None] - 1 - np.arange(20)[None, :]
    vals = []
    for values in (cosine, lift, latent):
        vals.extend([
            values.sum(axis=2) / np.maximum(length[:, None], 1),
            values.max(axis=2),
        ])
        for recent in (1, 3, 5):
            mask = valid[:, None, :] & (age[:, None, :] < recent)
            vals.append((values * mask).sum(axis=2)
                        / np.maximum(mask.sum(axis=2), 1))
    vals.extend([
        (c > 0).sum(axis=2),
        np.log1p(c.sum(axis=2)),
        np.log1p(c.max(axis=2)),
    ])
    extra = np.stack(vals, axis=2)
    rel = []
    for col in (0, 1, 5, 6, 10, 11):
        vv = extra[:, :, col]
        rel.extend([vv - vv.mean(axis=1, keepdims=True),
                    vv - vv.max(axis=1, keepdims=True)])
    return np.concatenate(
        [x, extra.reshape(-1, 18)] + [r.reshape(-1, 1) for r in rel], axis=1
    )


def _feature_context(ds, events, base):
    relations, context = base
    n = len(events)
    items = ds["items"][events]
    b = ds["tb"][events]
    users = ds["user"][events]
    g = context["global"]
    x = _feature_relations(ds, events, relations)

    cols = []
    for name, ids in (("user", users[:, None].repeat(10, 1)),
                      ("item", items)):
        key = ids * 4 + b[:, None]
        exposure = context[name + "_impr"][ids]
        clicks = context[name + "_click"][ids]
        prior = (clicks + 50.0 * g) / (exposure + 50.0)
        conditional = np.clip(
            prior * context["bucket_ctr"][b][:, None] / g, 1e-5, 0.999
        )
        ni_ = context[name + "_bucket_impr"][key]
        nc = context[name + "_bucket_click"][key]
        ri = context[name + "_recent_impr"][ids]
        rc = context[name + "_recent_click"][ids]
        cols.extend([
            (clicks + 10.0 * g) / (exposure + 10.0),
            (clicks + 150.0 * g) / (exposure + 150.0),
            (nc + 20.0 * conditional) / (ni_ + 20.0),
            (nc + 80.0 * conditional) / (ni_ + 80.0),
            np.log1p(ni_),
            (rc + 30.0 * prior) / (ri + 30.0),
            (rc + 30.0 * prior) / (ri + 30.0) - prior,
            ni_ / np.maximum(exposure, 1),
        ])
    cols.append(np.broadcast_to(context["bucket_ctr"][b][:, None], (n, 10)))
    extra = np.stack(cols, axis=2)

    rel = []
    for col in (8, 10, 11, 13, 14):
        vv = extra[:, :, col]
        rel.extend([vv - vv.mean(axis=1, keepdims=True),
                    vv - vv.max(axis=1, keepdims=True)])
    return np.concatenate(
        [x, extra.reshape(-1, 17)] + [r.reshape(-1, 1) for r in rel], axis=1
    )


def _feature_preference(ds, events, stats):
    base = stats[0]
    left, right, taste, profile, contrast, adjusted, pc = stats[1:]
    n = len(events)
    items = ds["items"][events]
    users = ds["user"][events]
    h = ds["hist"][events]
    valid = h >= 0
    safe = np.maximum(h, 0)
    length = valid.sum(axis=1)
    age = length[:, None] - 1 - np.arange(20)[None, :]

    x = _feature_context(ds, events, base)
    l = left[safe]
    r = right[items]
    vals = []
    for rank in (8, 16, 32, 64, 128):
        ls = l[:, :, 1:rank]
        rs = r[:, :, 1:rank]
        raw = np.einsum("nhd,ncd->nch", ls, rs) * valid[:, None, :]
        sim = raw / np.maximum(
            np.linalg.norm(ls, axis=2)[:, None, :]
            * np.linalg.norm(rs, axis=2)[:, :, None], 1e-8
        )
        vals.extend([
            raw.sum(axis=2) / np.maximum(length[:, None], 1),
            sim.sum(axis=2) / np.maximum(length[:, None], 1),
            sim.max(axis=2),
        ])
        for recent in (3, 5):
            mask = valid[:, None, :] & (age[:, None, :] < recent)
            vals.append((sim * mask).sum(axis=2)
                        / np.maximum(mask.sum(axis=2), 1))

    t = taste[items]
    p = profile[users]
    c = contrast[users]
    ad = adjusted[users]
    vals.extend([
        (p[:, None, :] * t).sum(axis=2),
        (c[:, None, :] * t).sum(axis=2),
        (ad[:, None, :] * t).sum(axis=2),
        np.log1p(pc[users])[:, None].repeat(10, 1),
    ])
    ht = (taste[safe] * valid[:, :, None]).sum(axis=1) / np.maximum(
        length[:, None], 1
    )
    ht = ht / np.maximum(np.linalg.norm(ht, axis=1, keepdims=True), 1e-8)
    vals.extend([
        (ht[:, None, :] * t).sum(axis=2),
        (ht * p).sum(axis=1)[:, None].repeat(10, 1),
    ])
    extra = np.stack(vals, axis=2)
    extra = np.concatenate([
        extra,
        p[:, None, :16] * t[:, :, :16],
        ht[:, None, :16] * t[:, :, :16],
    ], axis=2)

    rel = []
    for col in (1, 6, 11, 16, 21, 25, 26, 27, 29):
        vv = extra[:, :, col]
        rel.extend([vv - vv.mean(axis=1, keepdims=True),
                    vv - vv.max(axis=1, keepdims=True)])
    return np.concatenate(
        [x, extra.reshape(-1, 63)] + [r.reshape(-1, 1) for r in rel], axis=1
    ).astype(np.float32)


def _context_codes(stamps):
    return stamps % 86400 // 21600, (((stamps // 86400 + 3) % 7) >= 5).astype(np.int32)


def get_context_counts(ds, nu, ni, cutoff):
    path = FEAT / f"ctxcounts_{cutoff}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    log(f"[context] counts cutoff={cutoff}")
    idx = np.flatnonzero(ds["ts"] < cutoff)
    _, first = np.unique(ds["user"][idx], return_index=True)
    events = idx[first]
    h = ds["hist"][events]
    ts = ds["hts"][events]
    valid = h >= 0
    users = np.broadcast_to(ds["user"][events, None], h.shape)
    records = np.rec.fromarrays(
        [users[valid], h[valid], ts[valid]], names="user,item,timestamp"
    )
    _, unique = np.unique(records, return_index=True)
    wi = h[valid][unique]
    wt = ts[valid][unique]

    ce, ci = _clicked_events(ds, idx)
    ids = np.concatenate([wi, ci]).astype(np.int64)
    stamps = np.concatenate([wt, ds["ts"][ce]]).astype(np.int64)
    bucket, kind = _context_codes(stamps)
    joint = bucket * 2 + kind
    total = np.bincount(ids, minlength=ni)
    byjoint = np.bincount(ids * 8 + joint, minlength=ni * 8).reshape(ni, 4, 2)
    globaljoint = (byjoint.sum(axis=0) + 1.0) / (len(ids) + 8.0)
    result = dict(
        total=total,
        bucket=byjoint.sum(axis=2),
        kind=byjoint.sum(axis=1),
        joint=byjoint,
        globaljoint=globaljoint,
        globalbucket=globaljoint.sum(axis=1),
        globalkind=globaljoint.sum(axis=0),
    )
    np.savez_compressed(path, **result)
    return result


def _feature_context_prior(ds, events, counts):
    n = len(events)
    items = ds["items"][events]
    e_users = ds["user"][events]
    b = ds["tb"][events]
    k = ds["dt"][events]
    h = ds["hist"][events]
    ts = ds["hts"][events]
    valid = h >= 0
    hb, hk = _context_codes(ts)

    gb = counts["globalbucket"][b]
    gk = counts["globalkind"][k]
    gj = counts["globaljoint"][b, k]
    total = counts["total"][items]
    cb = counts["bucket"][items, b[:, None]]
    ck = counts["kind"][items, k[:, None]]
    cj = counts["joint"][items, b[:, None], k[:, None]]

    cols = [np.log1p(total), np.log1p(cb), np.log1p(ck), np.log1p(cj)]
    for alpha in (5.0, 20.0, 100.0):
        cols.extend([
            np.log((cb + alpha * gb[:, None]) / (total + alpha) / gb[:, None]),
            np.log((ck + alpha * gk[:, None]) / (total + alpha) / gk[:, None]),
            np.log((cj + alpha * gj[:, None]) / (total + alpha)
                   / gj[:, None]),
        ])
    eb = b
    ek = k
    user_counts = [
        ((hb == eb[:, None]) & valid).sum(axis=1),
        ((hk == ek[:, None]) & valid).sum(axis=1),
        ((hb == eb[:, None]) & (hk == ek[:, None]) & valid).sum(axis=1),
    ]
    priors = [
        counts["globalbucket"][eb],
        counts["globalkind"][ek],
        counts["globaljoint"][eb, ek],
    ]
    cnt = valid.sum(axis=1)
    for c, p in zip(user_counts, priors):
        cols.append(
            np.broadcast_to(
                (np.log((c + 5.0 * p) / (cnt + 5.0) / p))[:, None], (n, 10)
            )
        )
    raw = np.stack(cols, axis=2)
    rel = []
    for col in (0, 1, 3, 4, 5, 6, 7, 9):
        vv = raw[:, :, col]
        rel.extend([vv - vv.mean(axis=1, keepdims=True),
                    vv - vv.max(axis=1, keepdims=True)])
    return np.concatenate(
        [raw.reshape(-1, 16)] + [r.reshape(-1, 1) for r in rel], axis=1
    ).astype(np.float32)


def _warm_component_features(ds, events, cutoff, start, nu, ni, train_ds=None):
    """4 warm scores + 32 decomposition + 32 aligned factor interactions."""
    train_ds = ds if train_ds is None else train_ds
    U, V, H, bu, bi = get_warm_model(train_ds, nu, ni, start, cutoff)
    anchor = get_warm_model(train_ds, nu, ni, start, start + 2 * DAY)[1].astype(np.float64)
    current = V.astype(np.float64)
    anchor /= np.maximum(np.linalg.norm(anchor, axis=1, keepdims=True), 1e-12)
    current /= np.maximum(np.linalg.norm(current, axis=1, keepdims=True), 1e-12)
    left, _, right = np.linalg.svd(current.T @ anchor, full_matrices=False)
    rotation = left @ right
    assert np.max(np.abs(rotation.T @ rotation - np.eye(16))) < 1e-10

    ur = U.astype(np.float64) @ rotation
    vr = V.astype(np.float64) @ rotation
    hr = H.astype(np.float64) @ rotation
    ur = (ur / np.maximum(np.linalg.norm(ur, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    vr = (vr / np.maximum(np.linalg.norm(vr, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    hr = hr.astype(np.float32)

    U = U.astype(np.float32)
    V = V.astype(np.float32)
    H = H.astype(np.float32)
    bu = bu.astype(np.float32)
    bi = bi.astype(np.float32)

    out = np.empty((len(events) * 10, 68), dtype=np.float32)
    chunk = 20000
    for offset in range(0, len(events), chunk):
        e = events[offset:offset + chunk]
        m = len(e)
        items = ds["items"][e]
        users = ds["user"][e]
        h = ds["hist"][e]
        valid = h >= 0
        safe = np.maximum(h, 0)
        length = valid.sum(axis=1)
        age = length[:, None] - 1 - np.arange(20)[None, :]

        raw = _predict_events(users, items, h, U, V, H, bu, bi)
        flat = raw.ravel()
        rel = (raw - raw.mean(axis=1, keepdims=True)).ravel()
        rank = (np.argsort(np.argsort(raw, axis=1), axis=1) / 9.0).ravel()
        warm = np.stack([
            flat,
            np.log(np.clip(flat, 1e-8, 1 - 1e-8))
            - np.log1p(-np.clip(flat, 1e-8, 1 - 1e-8)),
            rel,
            rank,
        ], axis=1)

        hv = H[safe] * valid[:, :, None]
        z = hv.sum(axis=1) / np.sqrt(np.maximum(length[:, None], 1))
        u = U[users]
        v = V[items]
        direct = np.einsum("ed,ecd->ec", u, v)
        dynamic = np.einsum("ed,ecd->ec", z, v)
        un = np.linalg.norm(u, axis=1)
        zn = np.linalg.norm(z, axis=1)
        vn = np.linalg.norm(v, axis=2)
        sim = np.einsum("ehd,ecd->ech", hv, v) * valid[:, None, :]
        mean = sim.sum(axis=2) / np.maximum(length[:, None], 1)
        maximum = np.where(valid[:, None, :], sim, -1e6).max(axis=2)
        maximum[length == 0] = 0.0
        variance = np.maximum(
            (sim * sim).sum(axis=2) / np.maximum(length[:, None], 1) - mean * mean,
            0.0,
        )
        recents = []
        for width in (3, 5):
            mask = valid & (age < width)
            recents.append((sim * mask[:, None, :]).sum(axis=2)
                           / np.maximum(mask.sum(axis=1)[:, None], 1))
        agreement = (u * z).sum(axis=1) / np.maximum(un * zn, 1e-8)
        user_bias = np.broadcast_to(bu[users][:, None], direct.shape)
        item_bias = bi[items]
        logits = np.clip(-1.8 + user_bias + item_bias + direct + dynamic, -30, 30)
        ml = np.maximum(logits.max(axis=1, keepdims=True), 0.0)
        exp = np.exp(logits - ml)
        prob = exp / (np.exp(-ml) + exp.sum(axis=1, keepdims=True))
        no_click = 1.0 - prob.sum(axis=1)
        values = [
            user_bias, item_bias, direct, dynamic,
            np.broadcast_to(un[:, None], direct.shape),
            np.broadcast_to(zn[:, None], direct.shape),
            vn,
            direct / np.maximum(un[:, None] * vn, 1e-8),
            dynamic / np.maximum(zn[:, None] * vn, 1e-8),
            mean, maximum, np.sqrt(variance), *recents,
            np.broadcast_to(agreement[:, None], direct.shape),
            np.broadcast_to(no_click[:, None], direct.shape),
            direct - dynamic,
            -1.8 + user_bias + item_bias,
        ]
        decomp = np.stack(values, axis=2)
        rel_cols = []
        for col in (1, 2, 3, 7, 8, 10, 12):
            vv = decomp[:, :, col]
            rel_cols.extend([vv - vv.mean(axis=1, keepdims=True),
                             vv - vv.max(axis=1, keepdims=True)])
        decomp = np.concatenate(
            [decomp] + [r[:, :, None] for r in rel_cols], axis=2
        ).reshape(-1, 32)

        z_al = hr[safe]
        z_al = (z_al * valid[:, :, None]).sum(axis=1)
        z_al = z_al / np.maximum(np.linalg.norm(z_al, axis=1, keepdims=True), 1e-12)
        v_al = vr[items]
        u_al = ur[users]
        aligned = np.concatenate([
            u_al[:, None, :] * v_al,
            z_al[:, None, :] * v_al,
        ], axis=2).reshape(-1, 32)

        row = slice(offset * 10, (offset + m) * 10)
        out[row, :4] = warm
        out[row, 4:36] = decomp
        out[row, 36:68] = aligned
    assert np.isfinite(out).all()
    return out


def _als_component_features(ds, events, cutoff, start, nu, ni, train_ds=None):
    """32 exposure-ALS summaries + 32 aligned ALS interactions."""
    train_ds = ds if train_ds is None else train_ds
    model = get_als_model(train_ds, nu, ni, start, cutoff)
    anchor = get_als_model(train_ds, nu, ni, start, start + 2 * DAY)
    V = model["V"].astype(np.float64)
    unit_V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    anchor_V = anchor["V"].astype(np.float64)
    anchor_V /= np.maximum(np.linalg.norm(anchor_V, axis=1, keepdims=True), 1e-12)
    left, _, right = np.linalg.svd(unit_V.T @ anchor_V, full_matrices=False)
    rotation = left @ right
    assert np.max(np.abs(rotation.T @ rotation - np.eye(16))) < 1e-10

    vr = V @ rotation
    vr /= np.maximum(np.linalg.norm(vr, axis=1, keepdims=True), 1e-12)
    vr = vr.astype(np.float32)
    vn_all = np.linalg.norm(model["V"], axis=1)
    item_exposures = model["item_exposures"]
    g = float(model["g"])

    out = np.empty((len(events) * 10, 64), dtype=np.float32)
    chunk = 20000
    for offset in range(0, len(events), chunk):
        e = events[offset:offset + chunk]
        m = len(e)
        items = ds["items"][e]
        h = ds["hist"][e]
        valid = h >= 0
        safe = np.maximum(h, 0)
        length = valid.sum(axis=1)
        age = length[:, None] - 1 - np.arange(20)[None, :]

        u, bias = _als_score(ds, e, model, nu, ni)
        v = model["V"][items]
        vn = vn_all[items]
        direct = np.einsum("ed,ecd->ec", u, v)
        profile = (model["V"][safe] * valid[:, :, None]).sum(axis=1) / np.maximum(
            length[:, None], 1
        )
        hn = np.linalg.norm(profile, axis=1)
        history_dot = np.einsum("ed,ecd->ec", profile, v)
        sim = np.einsum("ehd,ecd->ech", unit_V[safe], unit_V[items]) * valid[:, None, :]
        mean = sim.sum(axis=2) / np.maximum(length[:, None], 1)
        maximum = np.where(valid[:, None, :], sim, -1e6).max(axis=2)
        maximum[length == 0] = 0.0
        std = np.sqrt(np.maximum(
            (sim * sim).sum(axis=2) / np.maximum(length[:, None], 1) - mean * mean,
            0.0,
        ))
        recents = []
        for width in (3, 5):
            mask = valid & (age < width)
            recents.append((sim * mask[:, None, :]).sum(axis=2)
                           / np.maximum(mask.sum(axis=1)[:, None], 1))

        user_bias = np.broadcast_to(bias[:, None], direct.shape)
        values = [
            user_bias, model["bi"][items], direct,
            np.broadcast_to(np.linalg.norm(u, axis=1)[:, None], direct.shape),
            vn,
            direct / np.maximum(np.linalg.norm(u, axis=1)[:, None] * vn, 1e-12),
            g + user_bias + model["bi"][items] + direct,
            np.broadcast_to(np.log1p(model["exposures"][ds["user"][e]])[:, None],
                            direct.shape),
            history_dot,
            np.broadcast_to(hn[:, None], direct.shape),
            history_dot / np.maximum(hn[:, None] * vn, 1e-12),
            maximum, std, *recents,
            (h[:, None, :] == items[:, :, None]).sum(axis=2),
            np.broadcast_to(np.log1p(length)[:, None], direct.shape),
            np.log1p(item_exposures[items]),
        ]
        scalar = np.stack(values, axis=2)
        rel = []
        for col in (1, 2, 5, 6, 8, 10, 11):
            vv = scalar[:, :, col]
            rel.extend([vv - vv.mean(axis=1, keepdims=True),
                        vv - vv.max(axis=1, keepdims=True)])
        scalar = np.concatenate(
            [scalar] + [r[:, :, None] for r in rel], axis=2
        ).reshape(-1, 32)

        ur = u @ rotation
        ur /= np.maximum(np.linalg.norm(ur, axis=1, keepdims=True), 1e-12)
        hr = profile @ rotation
        hr /= np.maximum(np.linalg.norm(hr, axis=1, keepdims=True), 1e-12)
        aligned = np.concatenate([
            ur.astype(np.float32)[:, None, :] * vr[items],
            hr.astype(np.float32)[:, None, :] * vr[items],
        ], axis=2).reshape(-1, 32)

        row = slice(offset * 10, (offset + m) * 10)
        out[row, :32] = scalar
        out[row, 32:64] = aligned
    assert np.isfinite(out).all()
    return out


def build_block(stat_ds, feat_ds, nu, ni, start, name, events, cutoff, rebuild=False):
    path = FEAT / f"block_{name}_c{cutoff}_v{FEATURE_VERSION}.npy"
    expected = len(events) * 10
    if path.exists() and not rebuild:
        x = np.load(path, mmap_mode="r")
        if x.shape == (expected, N_FEATURES):
            log(f"[features] {name}: cached {x.shape}")
            return x
    log(f"[features] {name}: {len(events):,} events cutoff={cutoff}")
    t = time.time()
    stats = get_preference(stat_ds, nu, ni, cutoff)
    x = np.empty((expected, N_FEATURES), dtype=np.float32)
    x[:, :191] = _feature_preference(feat_ds, events, stats)
    log(f"[features] {name}: base 191 done {time.time() - t:.0f}s")
    x[:, 191:259] = _warm_component_features(
        feat_ds, events, cutoff, start, nu, ni, train_ds=stat_ds
    )
    log(f"[features] {name}: warm/decomp/aligned done {time.time() - t:.0f}s")
    counts = get_context_counts(stat_ds, nu, ni, cutoff)
    x[:, 259:291] = _feature_context_prior(feat_ds, events, counts)
    log(f"[features] {name}: context done {time.time() - t:.0f}s")
    x[:, 291:355] = _als_component_features(
        feat_ds, events, cutoff, start, nu, ni, train_ds=stat_ds
    )
    log(f"[features] {name}: ALS done {time.time() - t:.0f}s")
    np.save(path, x)
    log(f"[features] {name}: saved {x.shape} in {time.time() - t:.0f}s")
    return x


# ---------------------------------------------------------------------------
# Training / submission
# ---------------------------------------------------------------------------
def _day_events(ds, start, day, val_cut=None):
    lo = start + day * DAY
    hi = start + (day + 1) * DAY
    mask = (ds["ts"] >= lo) & (ds["ts"] < hi)
    if val_cut is not None:
        mask &= ds["ts"] < val_cut
    return np.flatnonzero(mask)


def _first_event_cohort(ds, start, day, val_cut=None):
    events = _day_events(ds, start, day, val_cut)
    _, first = np.unique(ds["user"][events], return_index=True)
    return events[first]


def _assemble(ds, blocks, path, rebuild=False):
    rows = sum(len(b["events"]) * 10 for b in blocks)
    if path.exists() and not rebuild:
        x = np.load(path, mmap_mode="r")
        y_path = path.parent / (path.stem + "_labels.npy")
        if y_path.exists() and x.shape == (rows, N_FEATURES):
            y = np.load(y_path)
            if x.shape[0] == len(y):
                return x, y
    x = np.empty((rows, N_FEATURES), dtype=np.float32)
    ys = []
    offset = 0
    for b in blocks:
        mm = build_block(ds, ds, b["nu"], b["ni"], b["start"], b["name"],
                         b["events"], b["cutoff"], rebuild=rebuild)
        n = len(b["events"]) * 10
        x[offset:offset + n] = mm
        ys.append(ds["labels"][b["events"]].ravel().astype(np.int8))
        offset += n
    y = np.concatenate(ys)
    np.save(path, x)
    np.save(path.parent / (path.stem + "_labels.npy"), y)
    return x, y


def train_lightgbm(x, y, xv, yv, num_threads=32, seed=42,
                   num_boost_round=1800, early_stopping=120, verbose=50):
    params = dict(
        objective="binary",
        metric="auc",
        num_leaves=63,
        learning_rate=0.03,
        min_data_in_leaf=100,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=2.0,
        max_bin=127,
        force_col_wise=True,
        deterministic=True,
        num_threads=num_threads,
        seed=seed,
        verbosity=-1,
    )
    dtrain = lgb.Dataset(x, label=y, free_raw_data=True)
    dvalid = lgb.Dataset(xv, label=yv, reference=dtrain, free_raw_data=True)
    callbacks = [lgb.log_evaluation(verbose)]
    if early_stopping:
        callbacks.append(lgb.early_stopping(early_stopping, first_metric_only=True))
    model = lgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        valid_sets=[dvalid], valid_names=["validation"], callbacks=callbacks,
    )
    trees = int(model.best_iteration or model.current_iteration())
    return model, trees


def _write_submission(rowids, pred, path):
    out = pd.DataFrame({"rowid": rowids, "predicted_probability": pred})
    assert len(out) == 529390
    assert out["rowid"].is_unique
    assert np.isfinite(out["predicted_probability"]).all()
    out.to_csv(path, index=False)


def train_ensemble(x, y, xv, yv, seeds, output_dir, num_threads,
                   rounds=None, prediction_x=None):
    """Fit seeds sequentially and average probabilities with bounded memory.

    Final fits reuse each seed's own validation-selected tree count. This keeps
    final inference consistent with the validation ensemble.
    """
    total = None
    tree_counts = []
    individual = {}
    stage = "final" if rounds is not None else "validation"
    for index, seed in enumerate(seeds):
        log(f"[{stage}] training seed={seed}")
        options = {} if rounds is None else dict(
            num_boost_round=rounds[index], early_stopping=0, verbose=200)
        model, trees = train_lightgbm(
            x, y, xv, yv, num_threads=num_threads, seed=seed, **options)
        scores = model.predict(xv if prediction_x is None else prediction_x,
                               num_iteration=trees)
        if not np.isfinite(scores).all():
            raise ValueError(f"Nonfinite predictions for seed {seed}")
        total = scores.copy() if total is None else total + scores
        tree_counts.append(trees)
        model.save_model(str(output_dir / f"gbdt_{stage}_seed{seed}.txt"),
                         num_iteration=trees)
        if rounds is None:
            individual[str(seed)] = float(roc_auc_score(yv, scores))
            log(f"[validation] seed={seed} AUC={individual[str(seed)]:.6f} trees={trees}")
        del model, scores
        gc.collect()
    return total / len(seeds), tree_counts, individual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-only", action="store_true",
                    help="train/evaluate only; do not fit final model or write submission")
    ap.add_argument("--rebuild-data", action="store_true")
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--train-days", default="3,4,5")
    ap.add_argument("--final-days", default="4,5,6")
    ap.add_argument("--threads", type=int, default=min(32, numba_config.NUMBA_NUM_THREADS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", help="comma-separated ensemble seeds; default: --seed and next two")
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/gbdt_position_v5"))
    ap.add_argument("--smoke", type=int, default=0,
                    help="limit events per block for a quick shape/runtime check")
    args = ap.parse_args()
    if not 1 <= args.threads <= numba_config.NUMBA_NUM_THREADS:
        ap.error(f"--threads must be between 1 and {numba_config.NUMBA_NUM_THREADS}")
    try:
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(range(args.seed, args.seed + 3))
    except ValueError:
        ap.error("--seeds must be comma-separated integers")
    if len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**31 for s in seeds):
        ap.error("seeds must be distinct integers between 0 and 2147483647")
    set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tr, ev, nu, ni, start, rowids, item_vals = prepare_data(args.rebuild_data)
    ds = tr
    val_cut = int(np.quantile(ds["ts"], 0.85))
    log(f"[data] validation cutoff={val_cut} "
        f"({pd.to_datetime(val_cut, unit='s')})")

    train_days = [int(x) for x in args.train_days.split(",") if x != ""]
    final_days = [int(x) for x in args.final_days.split(",") if x != ""]

    def block(name, events, cutoff, smoke=True):
        if args.smoke and len(events) > args.smoke and name != "eval":
            events = events[: args.smoke]
        return dict(name=name, events=events, cutoff=cutoff,
                    nu=nu, ni=ni, start=start)

    val_events = np.flatnonzero(ds["ts"] >= val_cut)
    val_blocks = [block("val", val_events, val_cut, smoke=True)]
    train_blocks = [
        block(f"day{d}", _day_events(ds, start, d, val_cut if d == 5 else None),
              start + d * DAY)
        for d in train_days
    ]

    x, y = _assemble(ds, train_blocks, FEAT / "train_matrix.npy",
                     args.rebuild_features)
    xv, yv = _assemble(ds, val_blocks, FEAT / "val_matrix.npy", args.rebuild_features)
    val_used = val_blocks[0]["events"]
    log(f"[train] X={x.shape} pos_rate={y.mean():.5f} "
        f"val={xv.shape} val_pos={yv.mean():.5f}")

    pred, tree_counts, individual = train_ensemble(
        x, y, xv, yv, seeds, args.output_dir, args.threads)
    auc = float(roc_auc_score(yv, pred))
    log(f"[RESULT] full temporal holdout AUC = {auc:.5f} "
        f"(seeds={seeds} trees={tree_counts})")
    cohort_auc = None

    # Reference-comparable first-event-per-user day-6 cohort.
    try:
        cohort = _first_event_cohort(ds, start, 6)
        lookup = {int(e): i for i, e in enumerate(val_used)}
        if all(int(e) in lookup for e in cohort):
            rows = []
            for e in cohort:
                i = lookup[int(e)]
                rows.extend(range(i * 10, i * 10 + 10))
            rows = np.asarray(rows, dtype=np.int64)
            cohort_auc = float(roc_auc_score(yv[rows], pred[rows]))
            log(f"[RESULT] day-6 first-event cohort AUC = "
                f"{cohort_auc:.5f} "
                f"({len(cohort):,} events)")
    except Exception as exc:
        log(f"[metric] cohort check skipped: {exc}")

    val_df = pd.DataFrame({
        "event_row": np.repeat(np.arange(len(val_used)), 10),
        "item_id": np.asarray(item_vals)[ds["items"][val_used].ravel()],
        "label": yv,
        "pred": pred,
    })
    val_path = args.output_dir / "val_preds_gbdt.parquet"
    val_df.to_parquet(val_path, index=False)
    log(f"[val] predictions written {val_path}")
    report = dict(feature_version=FEATURE_VERSION, features=N_FEATURES,
                  seeds=seeds, trees=tree_counts, individual_auc=individual,
                  validation_auc=auc, day6_first_event_auc=cohort_auc,
                  validation_cutoff=val_cut, train_days=train_days,
                  final_days=final_days, smoke=args.smoke,
                  feature_cache=str(FEAT), final_complete=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    if args.val_only:
        log(f"[done] {time.time() - t0:.0f}s")
        return

    del x, y, val_df
    gc.collect()

    # Final model: days 4-6, evaluation features cut off at day 7.
    final_blocks = [
        block(f"day{d}full", _day_events(ds, start, d), start + d * DAY)
        for d in final_days
    ]
    xf, yf = _assemble(ds, final_blocks, FEAT / "final_matrix.npy",
                       args.rebuild_features)
    log(f"[final] train rows={len(yf):,} trees={tree_counts}")
    eval_events = np.arange(len(ev["ts"]))
    xe = build_block(ds, ev, nu, ni, start, "eval", eval_events,
                     start + 7 * DAY, rebuild=args.rebuild_features)
    prob, _, _ = train_ensemble(
        xf, yf, xv, yv, seeds, args.output_dir, args.threads,
        rounds=tree_counts, prediction_x=xe)
    submission_path = args.output_dir / "submission_gbdt.csv"
    _write_submission(rowids, prob, submission_path)
    report.update(final_complete=True, elapsed_seconds=time.time() - t0,
                  submission=str(submission_path))
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    log(f"[final] {submission_path} written; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
