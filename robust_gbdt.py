# robust_gbdt.py
import time
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

CACHE = Path("cache")
FEAT = CACHE / "features_gbdt_v5"

def main():
    print("[data] Loading cached feature matrices...")
    # Train (Days 3,4,5)
    xt = np.load(FEAT / "train_matrix.npy", mmap_mode="r")
    yt = np.load(FEAT / "train_matrix_labels.npy")

    # Val (Day 6)
    xv = np.load(FEAT / "val_matrix.npy", mmap_mode="r")
    yv = np.load(FEAT / "val_matrix_labels.npy")

    # Final (Days 1-6)
    xf = np.load(FEAT / "final_matrix.npy", mmap_mode="r")
    yf = np.load(FEAT / "final_matrix_labels.npy")

    # --- FEATURE SELECTION ---
    # Position is feature 13 and is included among the first 25 columns.
    # We drop the warm, context, ALS, and SVD features entirely.
    def safe_slice(x):
        x_arr = np.asarray(x, dtype=np.float32)
        return x_arr[:, :25]

    xt = safe_slice(xt)
    xv = safe_slice(xv)
    xf = safe_slice(xf)

    print(f"[data] Robust Train={xt.shape}, Val={xv.shape}, Final={xf.shape}")

    # --- TRAIN ON DAYS 3,4,5 ---
    print("[train] Training Robust LightGBM on Train...")
    params = dict(
        objective="binary", metric="auc", num_leaves=31,
        learning_rate=0.03, min_data_in_leaf=200, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
        max_bin=127, num_threads=16, seed=42, verbosity=-1
    )
    dtrain = lgb.Dataset(xt, label=yt)
    dval = lgb.Dataset(xv, label=yv, reference=dtrain)

    model = lgb.train(
        params, dtrain, num_boost_round=1000,
        valid_sets=[dval], valid_names=["validation"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    best_trees = model.best_iteration
    val_auc = roc_auc_score(yv, model.predict(xv, num_iteration=best_trees))
    print(f"\n[RESULT] Robust Local AUC = {val_auc:.5f} (trees={best_trees})")
    print("(Note: A lower local score here is EXPECTED and GOOD. It means we removed the overfitting.)")

    # --- RETRAIN ON DAYS 1-6 ---
    print(f"\n[train] Re-training Robust Model on Final (Days 1-6) with {best_trees} trees...")
    w_final = np.linspace(0.1, 2.0, len(yf)).astype(np.float32)

    dfinal = lgb.Dataset(xf, label=yf, weight=w_final)
    final_model = lgb.train(params, dfinal, num_boost_round=best_trees)

    # --- PREDICT EVAL ---
    print("[predict] Predicting eval...")
    xe = np.load(FEAT / "block_eval_c1737936000_v5.npy", mmap_mode="r")
    xe = safe_slice(xe)

    preds = final_model.predict(xe)
    ev_df = pd.read_pickle(CACHE / "eval.pkl")
    rowids = [x for sl in ev_df["rowids"].values for x in sl]

    pd.DataFrame({"rowid": rowids, "predicted_probability": preds}).to_csv("submission_robust.csv", index=False)
    print("[save] submission_robust.csv written.")

if __name__ == "__main__":
    main()
