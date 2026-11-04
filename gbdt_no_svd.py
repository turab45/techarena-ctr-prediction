# gbdt_no_svd.py
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
    xt = np.load(FEAT / "train_matrix.npy", mmap_mode="r")
    yt = np.load(FEAT / "train_matrix_labels.npy")
    xv = np.load(FEAT / "val_matrix.npy", mmap_mode="r")
    yv = np.load(FEAT / "val_matrix_labels.npy")
    xf = np.load(FEAT / "final_matrix.npy", mmap_mode="r")
    yf = np.load(FEAT / "final_matrix_labels.npy")

    # --- FEATURE SELECTION ---
    # We slice to keep:
    # 0:25 (Static item/user CTRs)
    # 191:355 (Warm-16, Context counts, ALS factors)
    # Position is feature 13, already included in the first 25 features.
    def safe_slice(x):
        x_arr = np.asarray(x, dtype=np.float32)
        return np.column_stack([x_arr[:, :25], x_arr[:, 191:355]])

    xt = safe_slice(xt)
    xv = safe_slice(xv)
    xf = safe_slice(xf)

    print(f"[data] No-SVD Train={xt.shape}, Val={xv.shape}, Final={xf.shape}")

    print("[train] Training No-SVD LightGBM...")
    params = dict(
        objective="binary", metric="auc", num_leaves=63,
        learning_rate=0.03, min_data_in_leaf=500, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
        max_bin=127, num_threads=16, seed=42, verbosity=-1
    )
    dtrain = lgb.Dataset(xt, label=yt)
    dval = lgb.Dataset(xv, label=yv, reference=dtrain)

    model = lgb.train(
        params, dtrain, num_boost_round=1200,
        valid_sets=[dval], valid_names=["validation"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )

    best_trees = model.best_iteration
    val_auc = roc_auc_score(yv, model.predict(xv, num_iteration=best_trees))
    print(f"\n[RESULT] No-SVD Local AUC = {val_auc:.5f} (trees={best_trees})")

    # Feature Importances
    imp = pd.DataFrame({
        'feature': model.feature_name(),
        'gain': model.feature_importance('gain')
    }).sort_values('gain', ascending=False)
    print("\n[imp] Top 10 features:")
    print(imp.head(10).to_string(index=False))

    print(f"\n[train] Re-training No-SVD Model on Final (Days 1-6) with {best_trees} trees...")
    # Weights: Linear from 0.1 to 2.0 across the final matrix
    w_final = np.linspace(0.1, 2.0, len(yf)).astype(np.float32)
    dfinal = lgb.Dataset(xf, label=yf, weight=w_final)
    final_model = lgb.train(params, dfinal, num_boost_round=best_trees)

    print("[predict] Predicting eval...")
    xe = np.load(FEAT / "block_eval_c1737936000_v5.npy", mmap_mode="r")
    xe = safe_slice(xe)

    preds = final_model.predict(xe)
    ev_df = pd.read_pickle(CACHE / "eval.pkl")
    rowids = [x for sl in ev_df["rowids"].values for x in sl]

    pd.DataFrame({"rowid": rowids, "predicted_probability": preds}).to_csv("submission_no_svd.csv", index=False)
    print("[save] submission_no_svd.csv written.")

if __name__ == "__main__":
    main()
