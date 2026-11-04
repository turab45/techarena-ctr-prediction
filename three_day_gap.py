# three_day_gap.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

CACHE = Path("cache")
FEAT = CACHE / "features_gbdt_v5"  # Using the v5 cache from strict_temporal_v3

def main():
    print("[data] Loading features...")
    # Load Day 3 (first 977,170 rows of train_matrix)
    d3_rows = 977170
    xt = np.asarray(np.load(FEAT / "train_matrix.npy", mmap_mode="r")[:d3_rows], dtype=np.float32)
    yt = np.load(FEAT / "train_matrix_labels.npy")[:d3_rows]
    
    # Load Day 6 (val_matrix)
    xv = np.asarray(np.load(FEAT / "val_matrix.npy", mmap_mode="r"), dtype=np.float32)
    yv = np.load(FEAT / "val_matrix_labels.npy")

    # Load drift order and drop the top 100 drifting features
    order = json.loads(Path("drop_order.json").read_text())["order"]
    drop_set = set(order[:100])
    cols = np.array([c for c in range(xt.shape[1]) if c not in drop_set])
    
    xt = xt[:, cols]
    xv = xv[:, cols]
    
    print(f"[data] Train (Day 3) X={xt.shape}, Val (Day 6) X={xv.shape}")

    print("[train] Training on Day 3, validating on Day 6 (3-day gap)...")
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
    print(f"\n[RESULT] Strict 3-Day Gap Local AUC = {val_auc:.5f} (trees={best_trees})")

if __name__ == "__main__":
    main()