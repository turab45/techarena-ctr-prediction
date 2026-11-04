# no_svd_ens.py
import time
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

CACHE = Path("cache")
FEAT = CACHE / "features_gbdt_v5"

def safe_slice(x):
    x_arr = np.asarray(x, dtype=np.float32)
    return np.column_stack([x_arr[:, :25], x_arr[:, 191:355]])

def main():
    print("[data] Loading cached feature matrices...")
    xt = np.load(FEAT / "train_matrix.npy", mmap_mode="r")
    yt = np.load(FEAT / "train_matrix_labels.npy")
    xv = np.load(FEAT / "val_matrix.npy", mmap_mode="r")
    yv = np.load(FEAT / "val_matrix_labels.npy")
    xf = np.load(FEAT / "final_matrix.npy", mmap_mode="r")
    yf = np.load(FEAT / "final_matrix_labels.npy")

    xt = safe_slice(xt)
    xv = safe_slice(xv)
    xf = safe_slice(xf)

    w_train = np.linspace(0.5, 2.0, len(yt)).astype(np.float32)
    w_final = np.linspace(0.1, 2.0, len(yf)).astype(np.float32)

    ev_df = pd.read_pickle(CACHE / "eval.pkl")
    rowids = [x for sl in ev_df["rowids"].values for x in sl]
    xe = safe_slice(np.load(FEAT / "block_eval_c1737936000_v5.npy", mmap_mode="r"))

    # ==========================================
    # 1. XGBOOST
    # ==========================================
    print("\n[XGBOOST] Training on Train...")
    dtrain = xgb.DMatrix(xt, label=yt, weight=w_train)
    dval = xgb.DMatrix(xv, label=yv)

    xgb_params = {
        "objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
        "max_depth": 7, "learning_rate": 0.03, "subsample": 0.8,
        "colsample_bytree": 0.7, "lambda": 2.0, "seed": 42, "nthread": 16
    }
    xgb_model = xgb.train(
        xgb_params, dtrain, num_boost_round=1500,
        evals=[(dval, 'val')], early_stopping_rounds=100, verbose_eval=200
    )
    xgb_auc = roc_auc_score(yv, xgb_model.predict(dval, iteration_range=(0, xgb_model.best_iteration)))
    print(f"[RESULT] No-SVD XGBoost AUC = {xgb_auc:.5f}")

    print(f"[XGBOOST] Re-training on Final with {xgb_model.best_iteration} trees...")
    dfinal = xgb.DMatrix(xf, label=yf, weight=w_final)
    xgb_final = xgb.train(xgb_params, dfinal, num_boost_round=xgb_model.best_iteration)
    xgb_preds = xgb_final.predict(xgb.DMatrix(xe))
    pd.DataFrame({"rowid": rowids, "predicted_probability": xgb_preds}).to_csv("submission_nosvd_xgb.csv", index=False)

    # ==========================================
    # 2. CATBOOST
    # ==========================================
    print("\n[CATBOOST] Training on Train...")
    cb_model = CatBoostClassifier(
        iterations=1200, learning_rate=0.04, depth=7,
        loss_function='Logloss', eval_metric='Logloss',
        random_seed=42, od_type='Iter', od_wait=100,
        task_type='CPU', bootstrap_type='Bernoulli', subsample=0.8, l2_leaf_reg=3.0, verbose=200
    )
    cb_model.fit(Pool(xt, label=yt, weight=w_train), eval_set=Pool(xv, label=yv), use_best_model=True)
    cb_auc = roc_auc_score(yv, cb_model.predict_proba(Pool(xv))[:, 1])
    print(f"[RESULT] No-SVD CatBoost AUC = {cb_auc:.5f}")

    print(f"[CATBOOST] Re-training on Final with {cb_model.get_best_iteration()} trees...")
    cb_final = CatBoostClassifier(
        iterations=cb_model.get_best_iteration(), learning_rate=0.04, depth=7,
        loss_function='Logloss', task_type='CPU', bootstrap_type='Bernoulli',
        subsample=0.8, l2_leaf_reg=3.0, verbose=0
    )
    cb_final.fit(Pool(xf, label=yf, weight=w_final))
    cb_preds = cb_final.predict_proba(Pool(xe))[:, 1]
    pd.DataFrame({"rowid": rowids, "predicted_probability": cb_preds}).to_csv("submission_nosvd_cb.csv", index=False)

    print("\n[save] Saved submission_nosvd_xgb.csv and submission_nosvd_cb.csv")

if __name__ == "__main__":
    main()
