# ensemble_teammate.py
import json
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from pathlib import Path

CACHE = Path("cache")

def main():
    print("[data] Loading cached pickle files...")
    train = pd.read_pickle(CACHE / "train.pkl")
    ev = pd.read_pickle(CACHE / "eval.pkl")
    
    users = {}; items = {}
    for df in (train, ev):
        for u in df['user_id']: 
            if u not in users: users[u] = len(users)
        for row in df['items']:
            for i in row:
                if i not in items: items[i] = len(items)
        for row in df['history_items']:
            for i in row:
                if i not in items: items[i] = len(items)
                
    nu, ni = len(users), len(items)
    start = int(train['timestamp'].min())
    DAY = 86400

    def convert(df, labelled=True):
        n = len(df)
        data = np.zeros((n*10, 7), dtype='int32')
        hist = np.full((n, 20), -1, dtype='int32')
        for e, row in enumerate(df.itertuples(index=False)):
            h = [items[x] for x in row.history_items if x in items]
            hist[e, :min(len(h),20)] = h[:20]
            for j, it in enumerate(row.items):
                data[e*10+j] = users[row.user_id], items[it], row.timestamp, row.time_bucket, row.day_type, row.labels[j] if labelled else -1, e
        return data, hist

    print("[data] Encoding arrays...")
    tr_data, tr_hist = convert(train, True)
    ev_data, ev_hist = convert(ev, False)

    def statistics(cutoff):
        a = tr_data[tr_data[:,2] < cutoff]
        y = a[:,5]
        g = float(y.mean())
        ui = np.bincount(a[:,0], minlength=nu); uc = np.bincount(a[:,0], weights=y, minlength=nu)
        ii = np.bincount(a[:,1], minlength=ni); ic = np.bincount(a[:,1], weights=y, minlength=ni)
        events = a[::10]
        _, first = np.unique(events[:,0], return_index=True)
        firstevents = events[first]
        h = tr_hist[firstevents[:,6]]
        hu = np.repeat(firstevents[:,0],20); hi = h.ravel(); valid = hi >= 0
        clicked = a[y == 1]
        matrix = coo_matrix((np.ones(int(valid.sum())+len(clicked),dtype='float32'),
            (np.concatenate([hu[valid],clicked[:,0]]),np.concatenate([hi[valid],clicked[:,1]]))),shape=(nu,ni)).tocsr()
        matrix.data = np.log1p(matrix.data)
        svd = TruncatedSVD(n_components=32,n_iter=4,random_state=42)
        svd.fit(matrix)
        emb = svd.components_.T.copy()
        emb /= np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-8)
        return g,ui,uc,ii,ic,emb

    def features(a, histories, stats):
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
        return x

    print("[train] Final training on Day 6 (Jan 26)...")
    stats = statistics(start + 6*DAY)
    fit = tr_data[tr_data[:,2] >= start+6*DAY]
    xf = features(fit, tr_hist, stats)
    yf = fit[:,5]

    rowids = [x for sl in ev["rowids"].values for x in sl]
    xe = features(ev_data, ev_hist, stats)

    # 1. CatBoost (Exact Teammate Config)
    print("[train] Training CatBoost...")
    cb = CatBoostClassifier(iterations=291, depth=7, learning_rate=.08, loss_function='Logloss', random_seed=42, verbose=0)
    cb.fit(xf, yf)
    cb_preds = cb.predict_proba(xe)[:,1]
    pd.DataFrame({"rowid": rowids, "predicted_probability": cb_preds}).to_csv("sub_tm_cb.csv", index=False)

    # 2. LightGBM
    print("[train] Training LightGBM...")
    lgb_params = dict(objective="binary", num_leaves=31, learning_rate=0.05, min_data_in_leaf=100, verbose=-1, seed=42)
    lgb_model = lgb.train(lgb_params, lgb.Dataset(xf, label=yf), num_boost_round=291)
    lgb_preds = lgb_model.predict(xe)
    pd.DataFrame({"rowid": rowids, "predicted_probability": lgb_preds}).to_csv("sub_tm_lgb.csv", index=False)

    # 3. XGBoost
    print("[train] Training XGBoost...")
    xgb_params = dict(objective='binary:logistic', max_depth=6, learning_rate=0.05, seed=42, nthread=16)
    xgb_model = xgb.train(xgb_params, xgb.DMatrix(xf, label=yf), num_boost_round=291)
    xgb_preds = xgb_model.predict(xgb.DMatrix(xe))
    pd.DataFrame({"rowid": rowids, "predicted_probability": xgb_preds}).to_csv("sub_tm_xgb.csv", index=False)

    print("[save] All 13-feature ensemble submissions written.")

if __name__ == "__main__":
    main()