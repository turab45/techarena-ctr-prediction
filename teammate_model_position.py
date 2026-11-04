"""Verified, standalone replication of Engr Riaz Ahmad Butt's baseline.

Reference (main branch, inspected 2026-11-04):
  https://github.com/Engr-Riaz-Ahmad-Butt/techarena-ctr-prediction/blob/main/prepare.py
  https://github.com/Engr-Riaz-Ahmad-Butt/techarena-ctr-prediction/blob/main/model.py

This combines the source preparation and modeling steps in one file. In particular,
it reads raw candidate CSV rows IN ORIGINAL ORDER (not pipeline.py's sorted cache),
assigns user/item IDs in the reference order, builds all 13 reference features,
and uses the reference CPU CatBoost configuration and fit/validation schedule.

Training/preparation procedures and feature formulas match the reference scripts.
Only paths/output names, diagnostic labels, and optional command-line switches differ.
No leaderboard upload is performed.

Run from your techarena-ctr-prediction checkout:
  uv run python teammate_model_verified.py --val-only
  uv run python teammate_model_verified.py --reuse-prepared --val-only
  uv run python teammate_model_verified.py --reuse-prepared

Optional environment overrides: CTR_DATA_DIR, CTR_TRAIN_CSV, CTR_EVALUATION_CSV,
CTR_ARTIFACT_DIR (default: artifacts/teammate_exact).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

DAY = 86400
TRAIN_ROWS = 6_868_240
EVAL_ROWS = 529_390


def _paths():
    base = Path(__file__).resolve().parent
    data = Path(os.environ.get("CTR_DATA_DIR", base / "data")).resolve()
    train_csv = Path(os.environ.get("CTR_TRAIN_CSV", data / "train.csv")).resolve()
    eval_csv = Path(os.environ.get("CTR_EVALUATION_CSV", data / "evaluation.csv")).resolve()
    root = Path(os.environ.get("CTR_ARTIFACT_DIR", base / "artifacts" / "teammate_exact")).resolve()
    return train_csv, eval_csv, root


def prepare(train_csv: Path, eval_csv: Path, root: Path):
    """Exactly follow teammate prepare.py: stream CSV, contiguous 10-row events."""
    users, items = {}, {}

    def encode(mapping, key):
        if key not in mapping:
            mapping[key] = len(mapping)
        return mapping[key]

    def convert(path, name, count, labelled):
        data = np.lib.format.open_memmap(root / (name + '.npy'), mode='w+', dtype='int32', shape=(count, 7))
        histories = np.lib.format.open_memmap(root / (name + '_hist.npy'), mode='w+', dtype='int32', shape=(count // 10, 20))
        histories[:] = -1
        previous = None
        start = time.time()
        with path.open(newline='', encoding='utf-8') as f:
            for n, row in enumerate(csv.DictReader(f)):
                if n >= count:
                    raise ValueError(f"{path}: more than expected {count:,} rows")
                event = n // 10
                uid, iid = encode(users, row['user_id']), encode(items, row['item_id'])
                if n % 10 == 0:
                    previous = row['event_id']
                    h = [encode(items, x) for x in json.loads(row['history_item_ids'])]
                    assert len(h) <= 20
                    histories[event, :len(h)] = h
                assert previous == row['event_id'], 'Unexpected event ordering'
                data[n] = uid, iid, int(row['timestamp']), int(row['time_bucket']), int(row['day_type']), int(row['clicked']) if labelled else -1, event
                if (n+1) % 1_000_000 == 0:
                    print(name, n+1, 'rows', round(time.time()-start), 'seconds', flush=True)
        if n + 1 != count:
            raise ValueError(f"{path}: expected {count:,} rows, got {n + 1:,}")
        data.flush()
        histories.flush()
        print(name, 'complete', n+1, flush=True)

    convert(train_csv, 'train', TRAIN_ROWS, True)
    convert(eval_csv, 'evaluation', EVAL_ROWS, False)
    (root / 'ids.json').write_text(json.dumps({'users': users, 'items': items}))
    print('Users:', len(users), 'Items:', len(items), flush=True)


def statistics(train, hist, nu, ni, cutoff):
    # Same implementation as reference model.py: first event per user as it
    # appears in the original CSV, and pre-cutoff labels only.
    a = np.asarray(train[train[:,2] < cutoff])
    y = a[:,5]
    g = float(y.mean())
    ui = np.bincount(a[:,0], minlength=nu); uc = np.bincount(a[:,0], weights=y, minlength=nu)
    ii = np.bincount(a[:,1], minlength=ni); ic = np.bincount(a[:,1], weights=y, minlength=ni)
    events = a[::10]
    _, first = np.unique(events[:,0], return_index=True)
    firstevents = events[first]
    h = np.asarray(hist[firstevents[:,6]])
    hu = np.repeat(firstevents[:,0],20); hi = h.ravel(); valid = hi >= 0
    clicked = a[y == 1]
    matrix = coo_matrix((np.ones(int(valid.sum())+len(clicked),dtype='float32'),
        (np.concatenate([hu[valid],clicked[:,0]]),np.concatenate([hi[valid],clicked[:,1]]))),shape=(nu,ni)).tocsr()
    matrix.data = np.log1p(matrix.data)
    if not np.isfinite(matrix.data).all():
        raise ValueError('[SVD] Input matrix contains NaN or infinity.')
    print(f'[SVD] shape={matrix.shape} nnz={matrix.nnz:,} '
          f'input_min={matrix.data.min():.6f} '
          f'input_max={matrix.data.max():.6f}', flush=True)
    svd = TruncatedSVD(n_components=32,n_iter=4,random_state=42)
    svd.fit(matrix)
    if not np.isfinite(svd.components_).all():
        raise ValueError('[SVD] Components contain NaN or infinity. '
                         'Numerical instability is confirmed; stop instead of training.')
    emb = svd.components_.T.copy()
    emb /= np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-8)
    if not np.isfinite(emb).all():
        raise ValueError('[SVD] Normalized embedding contains NaN or infinity.')
    print(f'[SVD] PASS: finite components and embeddings, '
          f'min={emb.min():.6f} max={emb.max():.6f}; '
          f'EVR={svd.explained_variance_ratio_.sum():.6f}', flush=True)
    print('Statistics:', len(a), 'rows; CTR:',g,'embedding ready',flush=True)
    return g,ui,uc,ii,ic,emb


def features(a, histories, stats):
    # EXACT 13 feature expressions from teammate model.py. No position feature.
    g,ui,uc,ii,ic,emb = stats
    x = np.empty((len(a),13), dtype='float32')
    for offset in range(0,len(a),100000):
        b=a[offset:offset+100000]; u,i=b[:,0],b[:,1]
        h=np.asarray(histories[b[:,6]]); valid=h>=0; safe=np.maximum(h,0)
        lengths=valid.sum(axis=1)
        v=emb[safe]*valid[:,:,None]
        avg=v.sum(axis=1)/np.maximum(lengths[:,None],1)
        similarity=(v*emb[i,None,:]).sum(axis=2)
        x[offset:offset+len(b)] = np.column_stack([
            (uc[u]+50*g)/(ui[u]+50), (ic[i]+50*g)/(ii[i]+50),
            np.log1p(ui[u]),np.log1p(ii[i]),b[:,3],b[:,4],lengths,
            (h==i[:,None]).sum(axis=1), (avg*emb[i]).sum(axis=1),
            similarity.max(axis=1), (similarity*np.arange(1,21)).sum(axis=1)/np.maximum((valid*np.arange(1,21)).sum(axis=1),1),
            (((ic[safe]+50*g)/(ii[safe]+50))*valid).sum(axis=1)/np.maximum(lengths,1),
            (b[:,2]%86400)/86400])
        if not np.isfinite(x[offset:offset+len(b)]).all():
            raise ValueError('[features] NaN/infinity in feature block, '
                             f'rows {offset}:{offset+len(b)}')
    return x


def train_and_validate(root: Path, evaluation_csv: Path, val_only: bool):
    train = np.load(root/'train.npy', mmap_mode='r')
    hist = np.load(root/'train_hist.npy', mmap_mode='r')
    evaluation = np.load(root/'evaluation.npy', mmap_mode='r')
    ehist = np.load(root/'evaluation_hist.npy', mmap_mode='r')
    ids = json.loads((root/'ids.json').read_text())
    nu, ni = len(ids['users']), len(ids['items'])
    del ids
    start = int(train[:,2].min())

    stats=statistics(train, hist, nu, ni, start+4*DAY)
    fit=np.asarray(train[(train[:,2]>=start+4*DAY)&(train[:,2]<start+6*DAY)])
    later=np.asarray(train[train[:,2]>=start+6*DAY])
    events=later[::10]
    _,first=np.unique(events[:,0],return_index=True)
    val=later[(first[:,None]*10+np.arange(10)).ravel()]
    print('Model fit rows:',len(fit),'Validation rows:',len(val),flush=True)
    xf=features(fit,hist,stats); xv=features(val,hist,stats)
    # Preserve CPU training and the exact teammate hyperparameters.
    model=CatBoostClassifier(iterations=350,depth=7,learning_rate=.08,loss_function='Logloss',eval_metric='AUC',thread_count=6,random_seed=42,verbose=50,allow_writing_files=False)
    model.fit(xf,fit[:,5],eval_set=(xv,val[:,5]),early_stopping_rounds=50)
    pred=model.predict_proba(xv)[:,1]
    if not np.isfinite(pred).all():
        raise ValueError('[validation] Nonfinite predictions')
    baseline=xv[:,0]*xv[:,1]
    report={'validation_rows':len(val),'baseline_auc':float(roc_auc_score(val[:,5],baseline)),'model_auc':float(roc_auc_score(val[:,5],pred)),'iterations':int(model.tree_count_)}
    (root/'validation.json').write_text(json.dumps(report,indent=2))
    print('[RESULT]',report,flush=True)
    if val_only:
        print('[done] Validation only; no submission file generated.',flush=True)
        return

    del xf,xv,fit,val,later,stats
    stats=statistics(train, hist, nu, ni, start+6*DAY)
    fit=np.asarray(train[train[:,2]>=start+6*DAY])
    xf=features(fit,hist,stats)
    final=CatBoostClassifier(iterations=report['iterations'],depth=7,learning_rate=.08,loss_function='Logloss',thread_count=6,random_seed=42,verbose=50,allow_writing_files=False)
    final.fit(xf,fit[:,5]); final.save_model(str(root/'model.cbm'))
    del xf,fit
    xe=features(evaluation,ehist,stats)
    prob=final.predict_proba(xe)[:,1]
    assert len(prob)==529390 and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    source=evaluation_csv
    seen=set()
    with source.open(newline='',encoding='utf-8') as src, (root/'submission.csv').open('w',newline='',encoding='utf-8') as dest:
        writer=csv.writer(dest);writer.writerow(['rowid','predicted_probability'])
        count=0
        for row,p in zip(csv.DictReader(src),prob,strict=True):
            assert row['rowid'] not in seen
            seen.add(row['rowid']);writer.writerow([row['rowid'],format(p,'.10g')]);count+=1
    assert count==529390
    print('Verified submission:',count,'unique rows',flush=True)
    print('[save]',root/'submission.csv',flush=True)


def main():
    parser = argparse.ArgumentParser(description='Faithful Engr Riaz 13-feature baseline replication')
    parser.add_argument('--val-only', action='store_true', help='Stop after validating; no final fit/predictions')
    parser.add_argument('--reuse-prepared', action='store_true', help='Reuse already prepared .npy and ids.json; avoid re-reading CSVs')
    args = parser.parse_args()
    print('[replica-check] Raw-CSV reference pipeline + SVD diagnostics; '
          f'val_only={args.val_only} reuse_prepared={args.reuse_prepared}', flush=True)

    train_csv, eval_csv, root = _paths()
    root.mkdir(parents=True, exist_ok=True)
    for p in (train_csv, eval_csv):
        if not p.is_file():
            parser.error(f'Missing {p}. Download/place the raw CSV in data/ first.')
    outputs = [root/'train.npy', root/'train_hist.npy', root/'evaluation.npy', root/'evaluation_hist.npy', root/'ids.json']
    if args.reuse_prepared:
        missing = [str(p) for p in outputs if not p.is_file()]
        if missing:
            parser.error('--reuse-prepared requires an earlier successful preparation. Missing: ' + ', '.join(missing))
        print('[prepare] Reusing existing converted arrays. Make sure input CSVs have not changed.', flush=True)
    else:
        print('[prepare] Streaming ORIGINAL raw CSV order, as teammate prepare.py does.',flush=True)
        prepare(train_csv, eval_csv, root)
    train_and_validate(root, eval_csv, args.val_only)


if __name__ == '__main__':
    main()
