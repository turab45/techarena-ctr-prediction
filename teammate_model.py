# teammate_model.py
import json
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from pathlib import Path

CACHE = Path("cache")

def parse_list(s):
    try: return json.loads(s)
    except: return []

def main():
    print("[data] Loading cached pickle files...")
    train = pd.read_pickle(CACHE / "train.pkl")
    ev = pd.read_pickle(CACHE / "eval.pkl")

    # Build vocab
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

    print("[train] Training 13-feature model...")
    stats = statistics(start + 4*DAY)

    fit = tr_data[(tr_data[:,2] >= start+4*DAY) & (tr_data[:,2] < start+6*DAY)]
    later = tr_data[tr_data[:,2] >= start+6*DAY]

    events = later[::10]
    _, first = np.unique(events[:,0], return_index=True)
    val = later[(first[:,None]*10 + np.arange(10)).ravel()]

    xf = features(fit, tr_hist, stats)
    xv = features(val, tr_hist, stats)

    model = CatBoostClassifier(iterations=350, depth=7, learning_rate=.08, loss_function='Logloss', eval_metric='AUC', thread_count=8, random_seed=42, verbose=50)
    model.fit(xf, fit[:,5], eval_set=(xv, val[:,5]), early_stopping_rounds=50)

    pred = model.predict_proba(xv)[:,1]
    print(f"\n[RESULT] Teammate Replication Local AUC = {roc_auc_score(val[:,5], pred):.5f}")

    print("[train] Final training on Day 6...")
    stats = statistics(start + 6*DAY)
    fit = tr_data[tr_data[:,2] >= start+6*DAY]
    xf = features(fit, tr_hist, stats)

    final = CatBoostClassifier(iterations=model.tree_count_, depth=7, learning_rate=.08, loss_function='Logloss', thread_count=8, random_seed=42, verbose=50)
    final.fit(xf, fit[:,5])

    print("[predict] Predicting eval...")
    xe = features(ev_data, ev_hist, stats)
    prob = final.predict_proba(xe)[:,1]

    rowids = [x for sl in ev["rowids"].values for x in sl]
    pd.DataFrame({"rowid": rowids, "predicted_probability": prob}).to_csv("submission_teammate_exact.csv", index=False)
    print("[save] submission_teammate_exact.csv written.")

if __name__ == "__main__":
    main()