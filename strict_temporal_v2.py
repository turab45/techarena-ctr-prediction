# strict_temporal_v2.py
import time
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

CACHE = Path("cache")
FEAT = CACHE / "features_gbdt_v5"

def main():
    print("[data] Loading cached matrices (Days 3,4,5 and Day 6)...")

    # Train matrix has Days 3,4,5. Day 5 is exactly the last 984,530 rows.
    x_tr_all = np.load(FEAT / "train_matrix.npy", mmap_mode="r")
    y_tr_all = np.load(FEAT / "train_matrix_labels.npy")

    xt = np.asarray(x_tr_all[-984530:], dtype=np.float32)
    yt = y_tr_all[-984530:]

    # Val matrix has Day 6
    xv = np.asarray(np.load(FEAT / "val_matrix.npy", mmap_mode="r"), dtype=np.float32)
    yv = np.load(FEAT / "val_matrix_labels.npy")

    print(f"[data] Train (Day 5) X={xt.shape}, Val (Day 6) X={xv.shape}")

    print("[train] Training on Day 5, validating on Day 6 (1-day gap)...")
    params = dict(
        objective="binary", metric="auc", num_leaves=63,
        learning_rate=0.03, min_data_in_leaf=100, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
        max_bin=127, num_threads=16, seed=42, verbosity=-1
    )

    dtrain = lgb.Dataset(xt, label=yt)
    dval = lgb.Dataset(xv, label=yv, reference=dtrain)

    model = lgb.train(
        params, dtrain, num_boost_round=1500,
        valid_sets=[dval], valid_names=["validation"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )

    best_trees = model.best_iteration
    val_auc = roc_auc_score(yv, model.predict(xv, num_iteration=best_trees))
    print(f"\n[RESULT] Strict 1-Day Gap Local AUC = {val_auc:.5f} (trees={best_trees})")

    # --- FINAL TRAINING ON DAYS 5+6 ---
    print(f"\n[train] Re-training on Days 5+6 with {best_trees} trees...")
    xf = np.concatenate([xt, xv], axis=0)
    yf = np.concatenate([yt, yv], axis=0)

    # Weight Day 6 higher (2x) than Day 5
    w_final = np.ones(len(yf), dtype=np.float32)
    w_final[:984530] = 1.0
    w_final[984530:] = 2.0

    dfinal = lgb.Dataset(xf, label=yf, weight=w_final)
    final_model = lgb.train(params, dfinal, num_boost_round=best_trees)

    # --- PREDICT EVAL ---
    print("[predict] Predicting eval...")
    xe = np.asarray(np.load(FEAT / "block_eval_c1737936000_v5.npy", mmap_mode="r"), dtype=np.float32)

    preds = final_model.predict(xe)
    ev_df = pd.read_pickle(CACHE / "eval.pkl")
    rowids = [x for sl in ev_df["rowids"].values for x in sl]

    pd.DataFrame({"rowid": rowids, "predicted_probability": preds}).to_csv("submission_strict_temporal.csv", index=False)
    print("[save] submission_strict_temporal.csv written.")

if __name__ == "__main__":
    main()
