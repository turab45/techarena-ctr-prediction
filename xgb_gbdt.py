# xgb_gbdt.py
import time
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import roc_auc_score

CACHE = Path("cache")
FEAT = CACHE / "features_gbdt_v5"

def main():
    print("[data] Loading cached feature matrices...")
    xf = np.load(FEAT / "final_matrix.npy", mmap_mode="r")
    yf = np.load(FEAT / "final_matrix_labels.npy")
    xv = np.load(FEAT / "val_matrix.npy", mmap_mode="r")
    yv = np.load(FEAT / "val_matrix_labels.npy")

    xf = np.asarray(xf, dtype=np.float32)
    xv = np.asarray(xv, dtype=np.float32)

    # Time weights for Final (Days 4,5,6)
    w_final = np.ones(len(yf), dtype=np.float32)
    w_final[:975890] = 0.5
    w_final[975890:975890+984530] = 1.0
    w_final[975890+984530:] = 2.0

    print("[train] Starting XGBoost on GPU...")
    dtrain = xgb.DMatrix(xf, label=yf, weight=w_final)
    dval = xgb.DMatrix(xv, label=yv)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda',  # Uses your RTX 4090
        'max_depth': 7,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'lambda': 2.0,
        'seed': 42
    }

    t0 = time.time()
    model = xgb.train(
        params, dtrain,
        num_boost_round=1500,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=200
    )

    print(f"[train] done in {time.time()-t0:.1f}s")

    preds_val = model.predict(dval, iteration_range=(0, model.best_iteration))
    auc = roc_auc_score(yv, preds_val)
    print(f"\n[RESULT] XGBoost Local AUC = {auc:.5f}\n")

    print("[predict] Loading eval features...")
    xe = np.load(FEAT / "block_eval_c1737936000_v5.npy", mmap_mode="r")
    xe = np.asarray(xe, dtype=np.float32)
    deval = xgb.DMatrix(xe)

    eval_preds = model.predict(deval, iteration_range=(0, model.best_iteration))

    ev_df = pd.read_pickle(CACHE / "eval.pkl")
    rowids = [x for sl in ev_df["rowids"].values for x in sl]

    pd.DataFrame({"rowid": rowids, "predicted_probability": eval_preds}).to_csv("submission_xgb.csv", index=False)
    print("[save] submission_xgb.csv written.")

if __name__ == "__main__":
    main()
