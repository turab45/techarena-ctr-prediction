# xgb_safe.py
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
    # Train (Days 3,4,5)
    xt = np.load(FEAT / "train_matrix.npy", mmap_mode="r")
    yt = np.load(FEAT / "train_matrix_labels.npy")
    xt = np.asarray(xt, dtype=np.float32)

    # Val (Day 6)
    xv = np.load(FEAT / "val_matrix.npy", mmap_mode="r")
    yv = np.load(FEAT / "val_matrix_labels.npy")
    xv = np.asarray(xv, dtype=np.float32)

    # Final (Days 1-6)
    xf = np.load(FEAT / "final_matrix.npy", mmap_mode="r")
    yf = np.load(FEAT / "final_matrix_labels.npy")
    xf = np.asarray(xf, dtype=np.float32)

    # Time weights (linear increase from 0.1 to 2.0 across the dataset)
    w_train = np.linspace(0.5, 2.0, len(yt)).astype(np.float32)
    w_final = np.linspace(0.1, 2.0, len(yf)).astype(np.float32)

    print("[train] Training XGBoost on Train (Days 3,4,5) to find trees...")
    dtrain = xgb.DMatrix(xt, label=yt, weight=w_train)
    dval = xgb.DMatrix(xv, label=yv)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'max_depth': 7,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'lambda': 2.0,
        'seed': 42,
        'nthread': 16
    }

    model = xgb.train(
        params, dtrain, num_boost_round=1500,
        evals=[(dval, 'val')], early_stopping_rounds=100, verbose_eval=200
    )

    best_trees = model.best_iteration
    preds_val = model.predict(dval, iteration_range=(0, best_trees))
    auc = roc_auc_score(yv, preds_val)
    print(f"\n[RESULT] XGBoost VALID Local AUC = {auc:.5f} (trees={best_trees})\n")

    print(f"[train] Re-training on Final (Days 1-6) with {best_trees} trees...")
    dfinal = xgb.DMatrix(xf, label=yf, weight=w_final)
    final_model = xgb.train(params, dfinal, num_boost_round=best_trees)

    print("[predict] Predicting eval...")
    xe = np.load(FEAT / "block_eval_c1737936000_v5.npy", mmap_mode="r")
    xe = np.asarray(xe, dtype=np.float32)
    deval = xgb.DMatrix(xe)

    preds = final_model.predict(deval)

    ev_df = pd.read_pickle(CACHE / "eval.pkl")
    rowids = [x for sl in ev_df["rowids"].values for x in sl]

    pd.DataFrame({"rowid": rowids, "predicted_probability": preds}).to_csv("submission_xgb_safe.csv", index=False)
    print("[save] submission_xgb_safe.csv written.")

if __name__ == "__main__":
    main()
