# catboost_gbdt.py
import time
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
import json

CACHE = Path("cache")
FEAT = CACHE / "features_gbdt_v4"

def main():
    print("[data] Loading cached feature matrices...")
    # The LightGBM run saved these exact files
    x = np.load(FEAT / "final_matrix.npy", mmap_mode="r")
    y = np.load(FEAT / "final_matrix_labels.npy")
    xv = np.load(FEAT / "val_matrix.npy", mmap_mode="r")
    yv = np.load(FEAT / "val_matrix_labels.npy")
    
    print(f"[data] Train X={x.shape}, Val X={xv.shape}")
    
    # CatBoost GPU requires float32
    x = np.asarray(x, dtype=np.float32)
    xv = np.asarray(xv, dtype=np.float32)

    train_pool = Pool(x, label=y)
    val_pool = Pool(xv, label=yv)

    print("[train] Starting CatBoost on GPU...")
    t0 = time.time()
    
    model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.04,
        depth=7,
        loss_function='Logloss',
        eval_metric='Logloss',       # AUC not implemented on GPU, use Logloss
        random_seed=42,
        od_type='Iter',
        od_wait=100,
        task_type='GPU',             # Uses your RTX 4090
        devices='0',
        bootstrap_type='Bernoulli',
        subsample=0.8,
        l2_leaf_reg=3.0,
        verbose=100
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    print(f"[train] done in {time.time()-t0:.1f}s")
    
    # Calculate validation AUC
    val_preds = model.predict_proba(val_pool)[:, 1]
    auc = roc_auc_score(yv, val_preds)
    print(f"\n[RESULT] CatBoost local holdout AUC = {auc:.5f}\n")
    
    # Save model
    model.save_model("catboost_model.cbm")

    # Predict Eval
    print("[predict] Loading eval features...")
    xe = np.load(FEAT / "block_eval_c1737936000_v4.npy", mmap_mode="r")
    xe = np.asarray(xe, dtype=np.float32)
    eval_pool = Pool(xe)
    
    prob = model.predict_proba(eval_pool)[:, 1]
    
    # Save submission
    ev_df = pd.read_pickle(CACHE / "eval.pkl")
    rowids = [x for sl in ev_df["rowids"].values for x in sl]
    
    sub = pd.DataFrame({"rowid": rowids, "predicted_probability": prob})
    assert len(sub) == 529390
    sub.to_csv("submission_catboost.csv", index=False)
    print("[save] submission_catboost.csv written.")

if __name__ == "__main__":
    main()