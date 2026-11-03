# pipeline.py
import argparse, json, math, pickle, time
import datetime as dt
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def parse_list(s):
    if not isinstance(s, str): return []
    try: return json.loads(s)
    except Exception: return []


def load_events(csv_path, cache_path, has_labels):
    if Path(cache_path).exists():
        print(f"[cache] {cache_path}")
        return pd.read_pickle(cache_path)
    print(f"[load] {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows  {df['event_id'].nunique():,} events")
    df['position'] = df.groupby('event_id', sort=False).cumcount()
    df = df.sort_values(['event_id', 'position'])

    agg = dict(
        user_id=('user_id', 'first'),
        timestamp=('timestamp', 'first'),
        time_bucket=('time_bucket', 'first'),
        day_type=('day_type', 'first'),
        history_str=('history_item_ids', 'first'),
        history_ts_str=('history_timestamps', 'first'),
        items=('item_id', list),
    )
    if has_labels: agg['labels'] = ('clicked', list)
    else:          agg['rowids'] = ('rowid', list)

    events = df.groupby('event_id', sort=False).agg(**agg).reset_index()
    events['history_items']      = events['history_str'].map(parse_list)
    events['history_timestamps'] = events['history_ts_str'].map(parse_list)
    events = events.drop(columns=['history_str', 'history_ts_str'])
    events.to_pickle(cache_path)
    return events


def build_vocab(*dfs):
    items, users = set(), set()
    for df in dfs:
        if df is None: continue
        for its in df['items']:            items.update(its)
        for hs  in df['history_items']:    items.update(hs)
        users.update(df['user_id'])
    item2idx = {v: i+1 for i, v in enumerate(sorted(items))}
    user2idx = {v: i+1 for i, v in enumerate(sorted(users))}
    return item2idx, user2idx


def compute_stats_from_events(events):
    """Compute item/user statistics from labeled fitting events only."""
    if 'labels' not in events.columns:
        raise ValueError("Need labeled events to compute CTR stats")
    item_clicks, item_n, item_last_seen = {}, {}, {}
    user_n_events, user_n_clicks = {}, {}
    global_clicks, global_n = 0, 0
    for r in events.itertuples(index=False):
        if len(r.items) != len(r.labels):
            raise ValueError("Candidate items and labels must have matching lengths")
        for item, label in zip(r.items, r.labels):
            label = int(label)
            item_clicks[item] = item_clicks.get(item, 0) + label
            item_n[item] = item_n.get(item, 0) + 1
            item_last_seen[item] = max(item_last_seen.get(item, r.timestamp), r.timestamp)
        user_n_events[r.user_id] = user_n_events.get(r.user_id, 0) + 1
        clicks = int(sum(r.labels))
        user_n_clicks[r.user_id] = user_n_clicks.get(r.user_id, 0) + clicks
        global_clicks += clicks
        global_n += len(r.labels)
    global_ctr = global_clicks / max(global_n, 1)
    item_stats = {
        item: ((item_clicks[item] + 10 * global_ctr) / (count + 10),
               count, item_last_seen[item])
        for item, count in item_n.items()
    }
    user_stats = {user: (count, user_n_clicks[user])
                  for user, count in user_n_events.items()}
    return item_stats, user_stats, global_ctr


def cmd_prep(args):
    tr  = load_events("data/train.csv",      CACHE/"train.pkl", True)
    ev  = load_events("data/evaluation.csv", CACHE/"eval.pkl",  False)
    i2i, u2i = build_vocab(tr, ev)
    with open(CACHE/"vocab.pkl", "wb") as f:
        pickle.dump({"item2idx": i2i, "user2idx": u2i}, f)
    print(f"[vocab] items={len(i2i):,} users={len(u2i):,}")
    # print time span so we pick a sensible split
    for name, df in [("train", tr), ("eval", ev)]:
        ts = df['timestamp']
        print(f"[span] {name}: {pd.to_datetime(ts.min(),unit='s')} → "
              f"{pd.to_datetime(ts.max(),unit='s')}  ({ts.nunique()} distinct ts)")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SlateDataset(Dataset):
    def __init__(self, events, item2idx, user2idx, max_hist=20, slate_size=10,
                 item_stats=None, user_stats=None, global_ctr=0.05):
        n = len(events); self.S = slate_size
        self.item2idx = item2idx
        self.training = False
        self.cand   = np.zeros((n, slate_size), dtype=np.int64)
        self.labels = np.zeros((n, slate_size), dtype=np.float32)
        self.hist   = np.zeros((n, max_hist),   dtype=np.int64)
        self.hist_len  = np.zeros(n, dtype=np.int64)
        self.hist_gap  = np.zeros((n, max_hist), dtype=np.float32)
        self.user      = np.zeros(n, dtype=np.int64)
        self.ts        = np.zeros(n, dtype=np.int64)
        self.tb        = np.zeros(n, dtype=np.int64)
        self.dt        = np.zeros(n, dtype=np.int64)
        self.in_hist   = np.zeros((n, slate_size), dtype=np.float32)
        self.item_ctr  = np.zeros((n, slate_size), dtype=np.float32)
        self.item_pop  = np.zeros((n, slate_size), dtype=np.float32)
        self.item_age  = np.zeros((n, slate_size), dtype=np.float32)
        self.user_act  = np.zeros(n, dtype=np.float32)
        self.user_ctr  = np.zeros(n, dtype=np.float32)
        self.hour_sin  = np.zeros(n, dtype=np.float32)
        self.hour_cos  = np.zeros(n, dtype=np.float32)
        self.dow_sin   = np.zeros(n, dtype=np.float32)
        self.dow_cos   = np.zeros(n, dtype=np.float32)
        has_lab = 'labels' in events.columns

        for i, r in enumerate(events.itertuples(index=False)):
            items = r.items[:slate_size]
            for j, it in enumerate(items):
                self.cand[i, j] = item2idx.get(it, 0)
                if item_stats is not None and it in item_stats:
                    ctr, count, last_seen = item_stats[it]
                    self.item_ctr[i, j] = ctr
                    self.item_pop[i, j] = math.log1p(count) / 15.0
                    age_hours = max(0.0, (r.timestamp - last_seen) / 3600.0)
                    self.item_age[i, j] = math.log1p(age_hours) / 10.0
            if has_lab:
                for j, l in enumerate(r.labels[:slate_size]):
                    self.labels[i, j] = float(l)
            hist  = r.history_items[-max_hist:]
            hts   = r.history_timestamps[-max_hist:]
            # Keep one unmasked slot to avoid NaNs in attention for empty histories.
            if len(hist) == 0:
                hist = [None]
                hts  = [r.timestamp]
            self.hist_len[i] = len(hist)
            for j, it in enumerate(hist):
                self.hist[i, j] = item2idx.get(it, 0)
            for j, t in enumerate(hts):
                self.hist_gap[i, j] = max(0.0, (r.timestamp - t) / 3600.0)
            cnts = {}
            for h in hist: cnts[h] = cnts.get(h, 0) + 1
            for j, it in enumerate(items):
                self.in_hist[i, j] = cnts.get(it, 0) / 20.0
            if user_stats is not None and r.user_id in user_stats:
                activity, clicks = user_stats[r.user_id]
                self.user_act[i] = math.log1p(activity) / 5.0
                self.user_ctr[i] = (clicks + 10 * global_ctr) / (activity * 10 + 10)
            timestamp = dt.datetime.fromtimestamp(int(r.timestamp), dt.timezone.utc)
            hour = timestamp.hour + timestamp.minute / 60.0
            self.hour_sin[i] = math.sin(2 * math.pi * hour / 24)
            self.hour_cos[i] = math.cos(2 * math.pi * hour / 24)
            self.dow_sin[i] = math.sin(2 * math.pi * timestamp.weekday() / 7)
            self.dow_cos[i] = math.cos(2 * math.pi * timestamp.weekday() / 7)
            self.user[i] = user2idx.get(r.user_id, 0)
            self.ts[i]   = r.timestamp
            self.tb[i]   = int(r.time_bucket)
            self.dt[i]   = int(r.day_type)
        # categorical sizes needed by model
        self.num_items = max(item2idx.values()) + 1
        self.num_users = max(user2idx.values()) + 1
        self.num_tb    = int(self.tb.max()) + 2
        self.num_dt    = int(self.dt.max()) + 2

    def __len__(self): return len(self.cand)

    def __getitem__(self, i):
        hist = self.hist[i].copy()
        hist_gap = self.hist_gap[i].copy()
        hist_len = int(self.hist_len[i])
        if self.training and hist_len > 5:
            keep = np.random.rand(hist_len) > 0.15
            keep[-1] = True
            n_keep = int(keep.sum())
            hist[:n_keep] = hist[:hist_len][keep]
            hist_gap[:n_keep] = hist_gap[:hist_len][keep]
            hist[n_keep:] = 0
            hist_gap[n_keep:] = 0
            hist_len = n_keep
        return dict(
            cand           = torch.from_numpy(self.cand[i]),
            labels         = torch.from_numpy(self.labels[i]),
            hist           = torch.from_numpy(hist),
            hist_len       = torch.tensor(hist_len, dtype=torch.long),
            hist_gap       = torch.from_numpy(hist_gap),
            in_hist        = torch.from_numpy(self.in_hist[i]),
            user           = torch.tensor(self.user[i], dtype=torch.long),
            ts             = torch.tensor(self.ts[i], dtype=torch.long),
            tb             = torch.tensor(self.tb[i], dtype=torch.long),
            dt             = torch.tensor(self.dt[i], dtype=torch.long),
            item_ctr       = torch.from_numpy(self.item_ctr[i]),
            item_pop       = torch.from_numpy(self.item_pop[i]),
            item_age       = torch.from_numpy(self.item_age[i]),
            user_act       = torch.tensor(self.user_act[i], dtype=torch.float32),
            user_ctr       = torch.tensor(self.user_ctr[i], dtype=torch.float32),
            hour_sin       = torch.tensor(self.hour_sin[i], dtype=torch.float32),
            hour_cos       = torch.tensor(self.hour_cos[i], dtype=torch.float32),
            dow_sin        = torch.tensor(self.dow_sin[i], dtype=torch.float32),
            dow_cos        = torch.tensor(self.dow_cos[i], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ModelCfg:
    num_items: int
    num_users: int
    num_tb: int
    num_dt: int
    d_model: int = 64
    n_heads: int = 4
    hist_layers: int = 1
    slate_layers: int = 1
    d_ff: int = 256
    dropout: float = 0.15
    max_hist: int = 20
    slate_size: int = 10
    gap_buckets: int = 64     # log-spaced hours
    numeric_features: bool = True


class Model(nn.Module):
    def __init__(self, cfg: ModelCfg, item2idx=None):
        super().__init__(); self.cfg = cfg
        d = cfg.d_model
        self.item_emb  = nn.Embedding(cfg.num_items, d, padding_idx=0)
        if item2idx is not None:
            with open(CACHE/"item2vec.pkl", "rb") as f:
                embeddings = pickle.load(f)
            n_loaded = 0
            with torch.no_grad():
                for item, vec in embeddings.items():
                    idx = item2idx.get(item)
                    if idx is None or idx == 0:
                        continue
                    vec = torch.as_tensor(vec, dtype=self.item_emb.weight.dtype)
                    if vec.ndim != 1 or vec.numel() == 0 or not torch.isfinite(vec).all():
                        raise ValueError(f"Invalid pretrained embedding for {item!r}")
                    d_use = min(d, vec.numel())
                    self.item_emb.weight[idx, :d_use].copy_(vec[:d_use])
                    n_loaded += 1
            if n_loaded == 0:
                raise ValueError("No pretrained item IDs match the training vocabulary")
            print(f"[model] loaded {n_loaded:,} / {cfg.num_items - 1:,} "
                  "pretrained item embeddings")
        self.user_emb  = nn.Embedding(cfg.num_users, d, padding_idx=0)
        self.pos_emb   = nn.Embedding(cfg.slate_size + 1, d)
        self.tb_emb    = nn.Embedding(cfg.num_tb, d)
        self.dt_emb    = nn.Embedding(cfg.num_dt, d)
        self.gap_emb   = nn.Embedding(cfg.gap_buckets, d)

        self.hist_pos  = nn.Embedding(cfg.max_hist, d)
        hist_layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, cfg.d_ff, cfg.dropout, batch_first=True, norm_first=True)
        self.hist_enc = nn.TransformerEncoder(hist_layer, cfg.hist_layers)

        slate_layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, cfg.d_ff, cfg.dropout, batch_first=True, norm_first=True)
        self.slate_enc = nn.TransformerEncoder(slate_layer, cfg.slate_layers)

        self.cross = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout,
                                           batch_first=True)
        self.cross_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d * 2 + (10 if cfg.numeric_features else 1), d),
            nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, 1),
        )

    def _gap_bucket(self, gap_hours):
        # log-spaced buckets: 0, then 1, 2, 4, 8, 16, ... hours up to ~1y
        edges = torch.tensor([0., 1., 2., 4., 8., 16., 32., 64., 128., 256.,
                              512., 1024., 2048., 4096., 8192.],
                             device=gap_hours.device)
        b = torch.bucketize(gap_hours, edges, right=False)
        return b.clamp_(0, self.cfg.gap_buckets - 1)

    def encode_history(self, hist, hist_len, hist_gap):
        B, L = hist.shape
        x = self.item_emb(hist)
        pos = torch.arange(L, device=hist.device).unsqueeze(0).expand(B, L)
        x = x + self.hist_pos(pos) + self.gap_emb(self._gap_bucket(hist_gap))
        pad = torch.arange(L, device=hist.device)[None, :] >= hist_len.clamp(min=1)[:, None]
        x = self.hist_enc(x, src_key_padding_mask=pad)
        return x, pad

    def forward(self, cand, hist, hist_len, hist_gap, in_hist,
                user, tb, dt, item_ctr, item_pop, item_age, user_act, user_ctr,
                hour_sin, hour_cos, dow_sin, dow_cos):
        B, S = cand.shape
        hist_enc, hist_pad = self.encode_history(hist, hist_len, hist_gap)

        c = self.item_emb(cand)
        pos = torch.arange(1, S + 1, device=cand.device).unsqueeze(0).expand(B, S)
        c = c + self.pos_emb(pos)
        c = c + self.user_emb(user).unsqueeze(1)
        c = c + self.tb_emb(tb).unsqueeze(1) + self.dt_emb(dt).unsqueeze(1)

        c = self.slate_enc(c)                      # in-slate competition
        attended, _ = self.cross(c, hist_enc, hist_enc,
                                 key_padding_mask=hist_pad, need_weights=False)
        c = self.cross_norm(c + attended)

        if self.cfg.numeric_features:
            cand_num = torch.stack([in_hist, item_ctr, item_pop, item_age], dim=-1)
            event_num = torch.stack(
                [user_act, user_ctr, hour_sin, hour_cos, dow_sin, dow_cos],
                dim=-1).unsqueeze(1).expand(-1, S, -1)
            feat = torch.cat([c, attended, cand_num, event_num], dim=-1)
        else:
            # Legacy checkpoints used unnormalised history counts and no numeric inputs.
            feat = torch.cat([c, attended, (in_hist * 20).unsqueeze(-1)], dim=-1)
        logits = self.head(feat).squeeze(-1)       # [B, S]
        return logits


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def listnet(logits, labels):
    mask = labels.sum(1) > 0
    if not mask.any(): return logits.new_zeros(())
    lp = F.log_softmax(logits[mask], dim=1)
    tgt = labels[mask].argmax(1)
    return -lp.gather(1, tgt.unsqueeze(1)).squeeze(1).mean()


def smooth_auc(logits, labels, tau=0.5):
    pairs = (labels == 1).unsqueeze(2) & (labels == 0).unsqueeze(1)
    pair_counts = pairs.sum(dim=(1, 2))
    losses = torch.sigmoid((logits.unsqueeze(1) - logits.unsqueeze(2)) / tau)
    per_slate = losses.masked_fill(~pairs, 0).sum(dim=(1, 2)) / pair_counts.clamp(min=1)
    return per_slate.sum() / (pair_counts > 0).sum().clamp(min=1)


def loss_fn(logits, labels, w_bce=1.0, w_ln=0.3, w_auc=0.1):
    bce = F.binary_cross_entropy_with_logits(logits, labels.float())
    return w_bce * bce + w_ln * listnet(logits, labels) + w_auc * smooth_auc(logits, labels)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def make_loaders(args):
    with open(CACHE/"vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    tr_all = pd.read_pickle(CACHE/"train.pkl")

    # temporal split — hold out last portion of days (default 15%)
    cut_ts = np.quantile(tr_all['timestamp'].values, 1 - args.val_frac)
    tr_df = tr_all[tr_all['timestamp'] <  cut_ts].reset_index(drop=True)
    va_df = tr_all[tr_all['timestamp'] >= cut_ts].reset_index(drop=True)
    print(f"[split] train={len(tr_df):,}  val={len(va_df):,}  cutoff={pd.to_datetime(cut_ts,unit='s')}")

    print("[stats] computing item/user CTRs from train portion only")
    item_stats, user_stats, global_ctr = compute_stats_from_events(tr_df)
    print(f"[stats] {len(item_stats):,} items, {len(user_stats):,} users, "
          f"global_ctr={global_ctr:.5f}")
    ds_tr = SlateDataset(tr_df, vocab['item2idx'], vocab['user2idx'],
                         args.max_hist, args.slate_size,
                         item_stats=item_stats, user_stats=user_stats, global_ctr=global_ctr)
    ds_va = SlateDataset(va_df, vocab['item2idx'], vocab['user2idx'],
                         args.max_hist, args.slate_size,
                         item_stats=item_stats, user_stats=user_stats, global_ctr=global_ctr)
    ds_tr.training = True
    ds_va.training = False
    print(f"[stats] feature ranges: "
          f"item_ctr=[0,{ds_tr.item_ctr.max():.2f}] "
          f"item_pop=[0,{ds_tr.item_pop.max():.2f}] "
          f"item_age=[0,{ds_tr.item_age.max():.2f}] "
          f"user_act=[0,{ds_tr.user_act.max():.2f}]")
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.bs * 2, shuffle=False,
                       num_workers=args.workers, pin_memory=True)
    return ds_tr, dl_tr, dl_va


def to_dev(batch, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


def forward(model, b):
    return model(b['cand'], b['hist'], b['hist_len'], b['hist_gap'],
                 b['in_hist'], b['user'], b['tb'], b['dt'],
                 b['item_ctr'], b['item_pop'], b['item_age'],
                 b['user_act'], b['user_ctr'],
                 b['hour_sin'], b['hour_cos'], b['dow_sin'], b['dow_cos'])


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); ys, ps = [], []
    for b in dl:
        b = to_dev(b, dev)
        logits = forward(model, b)
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(b['labels'].cpu().numpy())
    y = np.concatenate(ys).ravel(); p = np.concatenate(ps).ravel()
    return roc_auc_score(y, p)


def cmd_train(args):
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; GPU training cannot start. "
                         "Run 'uv run python test.py' to diagnose.")
    if dev == "cuda":
        print(f"[device] cuda:{torch.cuda.current_device()} — "
              f"{torch.cuda.get_device_name()}", flush=True)
    else:
        print("[device] cpu (explicitly requested)", flush=True)
    Path("ckpts").mkdir(exist_ok=True)
    if not (CACHE/"item2vec.pkl").is_file():
        raise SystemExit(f"Missing {CACHE/'item2vec.pkl'}. "
                         "Run 'uv run python pretrain_item2vec.py' first.")
    print("[data] preparing datasets on CPU before training", flush=True)
    ds_tr, dl_tr, dl_va = make_loaders(args)
    print(f"[diag] events with hist_len==0 after fix: "
          f"{int((ds_tr.hist_len == 0).sum())} train, "
          f"{int((dl_va.dataset.hist_len == 0).sum())} val")

    cfg = ModelCfg(
        num_items=ds_tr.num_items, num_users=ds_tr.num_users,
        num_tb=ds_tr.num_tb, num_dt=ds_tr.num_dt,
        d_model=args.d_model, n_heads=args.n_heads, dropout=args.dropout,
        max_hist=args.max_hist, slate_size=args.slate_size,
    )
    model = Model(cfg, item2idx=ds_tr.item2idx).to(dev)
    model.item_emb.requires_grad_(False)
    print("[freeze] item embeddings frozen for epochs 1–2")
    print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_steps = 500
    total_steps = args.epochs * len(dl_tr)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = GradScaler(device=dev)

    best = float('-inf')
    patience, bad = 3, 0
    step = 0
    for ep in range(1, args.epochs + 1):
        if ep == 3:
            model.item_emb.requires_grad_(True)
            print("[unfreeze] item embeddings now trainable")
        model.train(); t0 = time.time(); tot = 0.0
        for b in dl_tr:
            b = to_dev(b, dev)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=dev, dtype=torch.float16):
                logits = forward(model, b)
                loss = loss_fn(logits, b['labels'])
            if step == 0:
                print(f"[device] model={next(model.parameters()).device} "
                      f"batch={b['cand'].device} logits={logits.device}", flush=True)
            if not torch.isfinite(loss):
                print(f"[NaN] step {step}  loss={loss.item()}")
                raise SystemExit("Training diverged — aborting before checkpoint pollution")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            previous_scale = scaler.get_scale()
            scaler.step(opt); scaler.update()
            if scaler.get_scale() >= previous_scale:
                sched.step()
            tot += loss.item()
            step += 1
        auc = evaluate(model, dl_va, dev)
        print(f"[ep {ep:02d}] loss={tot/len(dl_tr):.4f}  val_auc={auc:.5f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}  "
              f"{time.time()-t0:.1f}s")
        torch.save({"model": model.state_dict(), "cfg": asdict(cfg),
                    "epoch": ep, "auc": auc},
                   f"ckpts/epoch_{ep:02d}_auc_{auc:.5f}.pt")
        if auc > best:
            best, bad = auc, 0
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg)},
                       "best_model.pt")
            print(f"   ↳ saved best  (best={best:.5f})")
        else:
            bad += 1
            if bad >= patience:
                print(f"[early-stop] no improvement for {patience} epochs")
                break
    print(f"[done] best val AUC = {best:.5f}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def cmd_predict(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("best_model.pt", map_location=dev)
    cfg = ModelCfg(**{"numeric_features": False, **ck["cfg"]})
    model = Model(cfg).to(dev); model.load_state_dict(ck["model"]); model.eval()

    with open(CACHE/"vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    ev = pd.read_pickle(CACHE/"eval.pkl")
    stats = {}
    if cfg.numeric_features:
        tr_full = pd.read_pickle(CACHE/"train.pkl")
        item_stats, user_stats, global_ctr = compute_stats_from_events(tr_full)
        stats = dict(item_stats=item_stats, user_stats=user_stats, global_ctr=global_ctr)
    ds = SlateDataset(ev, vocab['item2idx'], vocab['user2idx'],
                      cfg.max_hist, cfg.slate_size, **stats)
    dl = DataLoader(ds, batch_size=args.bs * 2, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    scores = []
    for b in dl:
        b = to_dev(b, dev)
        scores.append(torch.sigmoid(forward(model, b)).cpu().numpy())
    scores = np.concatenate(scores)                   # [N_events, 10]

    # flatten back to row order using eval.pkl rowids (already ordered by event/position)
    rows = []
    for i, r in enumerate(ev.itertuples(index=False)):
        for j, rid in enumerate(r.rowids):
            rows.append((rid, float(scores[i, j])))
    sub = pd.DataFrame(rows, columns=["rowid", "score"])
    sub.to_csv("submission_dl.csv", index=False)
    print(f"[predict] wrote submission_dl.csv  rows={len(sub)}")
    print(sub.head())


# ---------------------------------------------------------------------------
# Blend
# ---------------------------------------------------------------------------
def cmd_blend(args):
    """Rank-average two rowid-scored files (e.g. GBDT + transformer)."""
    a = pd.read_csv(args.a)[["rowid", "score"]].rename(columns={"score": "s_a"})
    b = pd.read_csv(args.b)[["rowid", "score"]].rename(columns={"score": "s_b"})
    m = a.merge(b, on="rowid")
    m["score"] = (args.wa * m["s_a"].rank(pct=True)
                  + (1 - args.wa) * m["s_b"].rank(pct=True))
    m[["rowid", "score"]].to_csv(args.out, index=False)
    print(f"[blend] wrote {args.out}  rows={len(m)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    sp.add_parser("prep")

    t = sp.add_parser("train")
    t.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                   help="training device (default: cuda; no automatic CPU fallback)")
    t.add_argument("--val_frac", type=float, default=0.15)
    t.add_argument("--bs", type=int, default=512)
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--d_model", type=int, default=64)
    t.add_argument("--n_heads", type=int, default=4)
    t.add_argument("--dropout", type=float, default=0.15)
    t.add_argument("--max_hist", type=int, default=20)
    t.add_argument("--slate_size", type=int, default=10)
    t.add_argument("--workers", type=int, default=8)

    pr = sp.add_parser("predict")
    pr.add_argument("--bs", type=int, default=1024)
    pr.add_argument("--workers", type=int, default=8)

    bl = sp.add_parser("blend")
    bl.add_argument("--a", required=True); bl.add_argument("--b", required=True)
    bl.add_argument("--out", required=True); bl.add_argument("--wa", type=float, default=0.5)

    args = p.parse_args()
    {"prep": cmd_prep, "train": cmd_train,
     "predict": cmd_predict, "blend": cmd_blend}[args.cmd](args)


if __name__ == "__main__":
    main()
