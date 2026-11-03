# train_time_weighted.py
import time
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import torch
from sklearn.metrics import roc_auc_score

CACHE = Path("cache")
FEAT = CACHE / "features_gbdt_v4"


def resolve_catboost_task_type():
    if torch.cuda.is_available():
        return "GPU", "0"
    print("[CATBOOST] CUDA is unavailable; falling back to CPU for CatBoost training.")
    return "CPU", None


def main():
    print("[data] Loading cached feature matrices...")
    x = np.load(FEAT / "final_matrix.npy", mmap_mode="r")
    y = np.load(FEAT / "final_matrix_labels.npy")
    xv = np.load(FEAT / "val_matrix.npy", mmap_mode="r")
    yv = np.load(FEAT / "val_matrix_labels.npy")
    
    x = np.asarray(x, dtype=np.float32)
    xv = np.asarray(xv, dtype=np.float32)

    # --- TIME DECAY WEIGHTS ---
    # Day 4 = 975,890 | Day 5 = 984,530 | Day 6 = 979,510
    weights = np.ones(len(y), dtype=np.float32)
    weights[:975890] = 0.5                          # Day 4
    weights[975890:975890+984530] = 1.0             # Day 5
    weights[975890+984530:] = 2.0                   # Day 6
    print(f"[weights] Applied time-decay weights (Day4=0.5, Day5=1.0, Day6=2.0)")

    # ---------------- 1. CATBOOST ----------------
    task_type, devices = resolve_catboost_task_type()
    print(f"\n[CATBOOST] Starting {task_type} training...")
    t0 = time.time()
    
    train_pool = Pool(x, label=y, weight=weights)
    val_pool = Pool(xv, label=yv)

    cb_kwargs = dict(
        iterations=1200,
        learning_rate=0.04,
        depth=7,
        loss_function='Logloss',
        eval_metric='Logloss',
        random_seed=42,
        od_type='Iter',
        od_wait=100,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        l2_leaf_reg=3.0,
        verbose=200,
        task_type=task_type,
    )
    if devices is not None:
        cb_kwargs['devices'] = devices

    cb_model = CatBoostClassifier(**cb_kwargs)

    cb_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    print(f"[CATBOOST] done in {time.time()-t0:.1f}s")
    
    cb_preds_val = cb_model.predict_proba(val_pool)[:, 1]
    cb_auc = roc_auc_score(yv, cb_preds_val)
    print(f"[RESULT] CatBoost Weighted Local AUC = {cb_auc:.5f}")

    # ---------------- 2. LIGHTGBM ----------------
    print("\n[LIGHTGBM] Starting training...")
    t0 = time.time()
    
    params = dict(
        objective="binary", metric="auc", num_leaves=63,
        learning_rate=0.03, min_data_in_leaf=100, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
        max_bin=127, num_threads=16, seed=42, verbosity=-1
    )
    
    dtrain = lgb.Dataset(x, label=y, weight=weights, free_raw_data=True)
    dvalid = lgb.Dataset(xv, label=yv, reference=dtrain, free_raw_data=True)
    
    lgb_model = lgb.train(
        params, dtrain, num_boost_round=1500,
        valid_sets=[dvalid], valid_names=["validation"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )
    
    lgb_preds_val = lgb_model.predict(xv, num_iteration=lgb_model.best_iteration)
    lgb_auc = roc_auc_score(yv, lgb_preds_val)
    print(f"[RESULT] LightGBM Weighted Local AUC = {lgb_auc:.5f}")

    # ---------------- 3. PREDICT EVAL ----------------
    print("\n[predict] Loading eval features...")
    xe = np.load(FEAT / "block_eval_c1737936000_v4.npy", mmap_mode="r")
    xe = np.asarray(xe, dtype=np.float32)
    eval_pool = Pool(xe)

    cb_eval = cb_model.predict_proba(eval_pool)[:, 1]
    lgb_eval = lgb_model.predict(xe, num_iteration=lgb_model.best_iteration)

    # Save both submissions
    ev_df = pd.read_pickle(CACHE / "eval.pkl")
    rowids = [x for sl in ev_df["rowids"].values for x in sl]
    
    pd.DataFrame({"rowid": rowids, "predicted_probability": cb_eval}).to_csv("submission_catboost_time.csv", index=False)
    pd.DataFrame({"rowid": rowids, "predicted_probability": lgb_eval}).to_csv("submission_gbdt_time.csv", index=False)
    
    print("[save] submission_catboost_time.csv and submission_gbdt_time.csv written.")

if __name__ == "__main__":
    main()