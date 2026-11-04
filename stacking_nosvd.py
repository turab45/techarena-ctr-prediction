# stacking_nosvd.py
import time
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

CACHE = Path("cache")
FEAT = CACHE / "features_gbdt_v5"

def safe_slice(x):
    x_arr = np.asarray(x, dtype=np.float32)
    return np.column_stack([x_arr[:, :25], x_arr[:, 191:355]])

def main():
    print("[data] Loading cached feature matrices...")
    xt = safe_slice(np.load(FEAT / "train_matrix.npy", mmap_mode="r"))
    yt = np.load(FEAT / "train_matrix_labels.npy")
    xv = safe_slice(np.load(FEAT / "val_matrix.npy", mmap_mode="r"))
    yv = np.load(FEAT / "val_matrix_labels.npy")
    xf = safe_slice(np.load(FEAT / "final_matrix.npy", mmap_mode="r"))
    yf = np.load(FEAT / "final_matrix_labels.npy")
    xe = safe_slice(np.load(FEAT / "block_eval_c1737936000_v5.npy", mmap_mode="r"))

    w_train = np.linspace(0.5, 2.0, len(yt)).astype(np.float32)
    w_final = np.linspace(0.1, 2.0, len(yf)).astype(np.float32)

    ev_df = pd.read_pickle(CACHE / "eval.pkl")
    rowids = [x for sl in ev_df["rowids"].values for x in sl]

    seeds = [42, 43, 44]
    val_preds = { 'lgb': [], 'xgb': [], 'cb': [] }
    eval_preds = { 'lgb': [], 'xgb': [], 'cb': [] }
    trees = { 'lgb': [], 'xgb': [], 'cb': [] }

    # ==========================================
    # 1. TRAIN BASE MODELS (3 SEEDS EACH)
    # ==========================================
    for seed in seeds:
        print(f"\n[SEED {seed}] Training LightGBM...")
        lgb_params = dict(
            objective="binary", metric="auc", num_leaves=63, learning_rate=0.03,
            min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=10.0, seed=seed, verbosity=-1
        )
        lgb_model = lgb.train(
            lgb_params, lgb.Dataset(xt, label=yt, weight=w_train),
            num_boost_round=1200, valid_sets=[lgb.Dataset(xv, label=yv)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        t = lgb_model.best_iteration
        trees['lgb'].append(t)
        val_preds['lgb'].append(lgb_model.predict(xv, num_iteration=t))

        print(f"[SEED {seed}] Training XGBoost...")
        xgb_params = {
            "objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
            "max_depth": 7, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7,
            "lambda": 2.0, "seed": seed, "nthread": 16,
        }
        xgb_model = xgb.train(
            xgb_params, xgb.DMatrix(xt, label=yt, weight=w_train),
            num_boost_round=1500, evals=[(xgb.DMatrix(xv, label=yv), 'val')],
            early_stopping_rounds=100, verbose_eval=False
        )
        t = xgb_model.best_iteration
        trees['xgb'].append(t)
        val_preds['xgb'].append(xgb_model.predict(xgb.DMatrix(xv), iteration_range=(0, t)))

        print(f"[SEED {seed}] Training CatBoost...")
        cb_model = CatBoostClassifier(
            iterations=1000, learning_rate=0.04, depth=7, loss_function='Logloss',
            eval_metric='Logloss', random_seed=seed, od_type='Iter', od_wait=50,
            task_type='CPU', bootstrap_type='Bernoulli', subsample=0.8, l2_leaf_reg=3.0, verbose=0
        )
        cb_model.fit(Pool(xt, label=yt, weight=w_train), eval_set=Pool(xv, label=yv), use_best_model=True)
        t = cb_model.get_best_iteration()
        trees['cb'].append(t)
        val_preds['cb'].append(cb_model.predict_proba(Pool(xv))[:, 1])

    # ==========================================
    # 2. AVERAGE PREDICTIONS & STACK
    # ==========================================
    print("\n[STACKING] Blending base models...")
    # Average across seeds
    lgb_val = np.mean(val_preds['lgb'], axis=0)
    xgb_val = np.mean(val_preds['xgb'], axis=0)
    cb_val = np.mean(val_preds['cb'], axis=0)

    print(f"LightGBM Averaged AUC: {roc_auc_score(yv, lgb_val):.5f}")
    print(f"XGBoost Averaged AUC:  {roc_auc_score(yv, xgb_val):.5f}")
    print(f"CatBoost Averaged AUC: {roc_auc_score(yv, cb_val):.5f}")

    # Stacking (Logistic Regression on the 3 predictions)
    X_meta_val = np.column_stack([lgb_val, xgb_val, cb_val])
    meta_model = LogisticRegression()
    meta_model.fit(X_meta_val, yv)

    stacked_val = meta_model.predict_proba(X_meta_val)[:, 1]
    stacked_auc = roc_auc_score(yv, stacked_val)
    print(f"\n[RESULT] STACKED Local AUC = {stacked_auc:.5f}")
    print(f"[meta] Weights: LGB={meta_model.coef_[0][0]:.3f}, XGB={meta_model.coef_[0][1]:.3f}, CB={meta_model.coef_[0][2]:.3f}")

    # ==========================================
    # 3. TRAIN ON FINAL & PREDICT EVAL
    # ==========================================
    print("\n[FINAL] Retraining base models on full data...")
    for i, seed in enumerate(seeds):
        # LightGBM
        lgb_model = lgb.train(dict(objective="binary", metric="auc", num_leaves=63, learning_rate=0.03,
            min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=10.0, seed=seed, verbosity=-1), lgb.Dataset(xf, label=yf, weight=w_final),
            num_boost_round=trees['lgb'][i])
        eval_preds['lgb'].append(lgb_model.predict(xe))

        # XGBoost
        xgb_model = xgb.train({
            "objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
            "max_depth": 7, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7,
            "lambda": 2.0, "seed": seed, "nthread": 16,
        }, xgb.DMatrix(xf, label=yf, weight=w_final),
            num_boost_round=trees['xgb'][i])
        eval_preds['xgb'].append(xgb_model.predict(xgb.DMatrix(xe)))

        # CatBoost
        cb_model = CatBoostClassifier(iterations=trees['cb'][i], learning_rate=0.04, depth=7,
            loss_function='Logloss', random_seed=seed, task_type='CPU', bootstrap_type='Bernoulli',
            subsample=0.8, l2_leaf_reg=3.0, verbose=0)
        cb_model.fit(Pool(xf, label=yf, weight=w_final))
        eval_preds['cb'].append(cb_model.predict_proba(Pool(xe))[:, 1])

    # Average eval predictions across seeds
    lgb_eval = np.mean(eval_preds['lgb'], axis=0)
    xgb_eval = np.mean(eval_preds['xgb'], axis=0)
    cb_eval = np.mean(eval_preds['cb'], axis=0)

    # Apply meta-model to eval
    X_meta_eval = np.column_stack([lgb_eval, xgb_eval, cb_eval])
    final_preds = meta_model.predict_proba(X_meta_eval)[:, 1]

    # Save
    pd.DataFrame({"rowid": rowids, "predicted_probability": final_preds}).to_csv("submission_stacked_nosvd.csv", index=False)
    print("\n[save] submission_stacked_nosvd.csv written.")

if __name__ == "__main__":
    main()
