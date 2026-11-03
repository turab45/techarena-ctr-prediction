# train_weighted_safe.py
import time
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

CACHE = Path("cache")
FEAT = CACHE / "features_gbdt_v4"

def main():
    print("[data] Loading cached feature matrices...")
    
    # 1. Load Valid Train Set (Days 3, 4, 5)
    xt = np.load(FEAT / "train_matrix.npy", mmap_mode="r")
    yt = np.load(FEAT / "train_matrix_labels.npy")
    xt = np.asarray(xt, dtype=np.float32)
    
    # 2. Load Validation Set
    xv = np.load(FEAT / "val_matrix.npy", mmap_mode="r")
    yv = np.load(FEAT / "val_matrix_labels.npy")
    xv = np.asarray(xv, dtype=np.float32)

    # 3. Load Final Train Set (Days 4, 5, 6)
    xf = np.load(FEAT / "final_matrix.npy", mmap_mode="r")
    yf = np.load(FEAT / "final_matrix_labels.npy")
    xf = np.asarray(xf, dtype=np.float32)

    # --- APPLY TIME WEIGHTS TO TRAIN ---
    # Train sizes: Day 3=97,717 (0.5), Day 4=97,589 (1.0), Day 5=93,370 (2.0)
    w_train = np.ones(len(yt), dtype=np.float32)
    w_train[:977170] = 0.5
    w_train[977170:977170+975890] = 1.0
    w_train[977170+975890:] = 2.0

    # --- APPLY TIME WEIGHTS TO FINAL ---
    # Final sizes: Day 4=97,589 (0.5), Day 5=98,453 (1.0), Day 6=97,951 (2.0)
    w_final = np.ones(len(yf), dtype=np.float32)
    w_final[:975890] = 0.5
    w_final[975890:975890+984530] = 1.0
    w_final[975890+984530:] = 2.0

    print(f"[weights] Applied. Train={xt.shape}, Final={xf.shape}")

    # ==========================================
    # 1. CATBOOST
    # ==========================================
    print("\n[CATBOOST] Training on Train (Days 3,4,5) to find trees...")
    train_pool = Pool(xt, label=yt, weight=w_train)
    val_pool = Pool(xv, label=yv)

    cb_model = CatBoostClassifier(
        iterations=1200, learning_rate=0.04, depth=7,
        loss_function='Logloss', eval_metric='Logloss',
        random_seed=42, od_type='Iter', od_wait=100,
        task_type='CPU', # CPU fallback since no GPU on mac
        bootstrap_type='Bernoulli', subsample=0.8,
        l2_leaf_reg=3.0, verbose=200
    )
    cb_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    cb_trees = cb_model.get_best_iteration()
    cb_auc = roc_auc_score(yv, cb_model.predict_proba(val_pool)[:, 1])
    print(f"[RESULT] CatBoost VALID Local AUC = {cb_auc:.5f} (trees={cb_trees})")

    print(f"\n[CATBOOST] Re-training on Final (Days 4,5,6) with {cb_trees} trees...")
    final_pool = Pool(xf, label=yf, weight=w_final)
    cb_final = CatBoostClassifier(
        iterations=cb_trees, learning_rate=0.04, depth=7,
        loss_function='Logloss', task_type='CPU',
        bootstrap_type='Bernoulli', subsample=0.8,
        l2_leaf_reg=3.0, verbose=0
    )
    cb_final.fit(final_pool)

    # ==========================================
    # 2. LIGHTGBM
    # ==========================================
    print("\n[LIGHTGBM] Training on Train (Days 3,4,5) to find trees...")
    params = dict(
        objective="binary", metric="auc", num_leaves=63,
        learning_rate=0.03, min_data_in_leaf=100, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
        max_bin=127, num_threads=16, seed=42, verbosity=-1
    )
    dtrain = lgb.Dataset(xt, label=yt, weight=w_train)
    dval = lgb.Dataset(xv, label=yv, reference=dtrain)
    
    lgb_model = lgb.train(
        params, dtrain, num_boost_round=1500,
        valid_sets=[dval], valid_names=["validation"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )
    
    lgb_trees = lgb_model.best_iteration
    lgb_auc = roc_auc_score(yv, lgb_model.predict(xv, num_iteration=lgb_trees))
    print(f"[RESULT] LightGBM VALID Local AUC = {lgb_auc:.5f} (trees={lgb_trees})")

    print(f"\n[LIGHTGBM] Re-training on Final (Days 4,5,6) with {lgb_trees} trees...")
    dfinal = lgb.Dataset(xf, label=yf, weight=w_final)
    lgb_final = lgb.train(params, dfinal, num_boost_round=lgb_trees)

    # ==========================================
    # 3. PREDICT & SAVE
    # ==========================================
    print("\n[predict] Loading eval features...")
    xe = np.load(FEAT / "block_eval_c1737936000_v4.npy", mmap_mode="r")
    xe = np.asarray(xe, dtype=np.float32)
    
    # Predict
    cb_eval = cb_final.predict_proba(Pool(xe))[:, 1]
    lgb_eval = lgb_final.predict(xe)

    ev_df = pd.read_pickle(CACHE / "eval.pkl")
    rowids = [x for sl in ev_df["rowids"].values for x in sl]
    
    pd.DataFrame({"rowid": rowids, "predicted_probability": cb_eval}).to_csv("submission_cb_safe.csv", index=False)
    pd.DataFrame({"rowid": rowids, "predicted_probability": lgb_eval}).to_csv("submission_lgb_safe.csv", index=False)
    
    print("[save] submission_cb_safe.csv and submission_lgb_safe.csv written.")

if __name__ == "__main__":
    main()