# diag.py
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def parse(s):
    try: return json.loads(s)
    except: return []

train = pd.read_csv("data/train.csv")
cut_ts = np.quantile(train['timestamp'].values, 0.85)
tr = train[train['timestamp'] <  cut_ts].copy()
va = train[train['timestamp'] >= cut_ts].copy()
print(f"tr={len(tr):,}  va={len(va):,}")

# --- 1. Item popularity (global CTR proxy) ---
item_ctr = tr.groupby('item_id')['clicked'].mean()
va['pop'] = va['item_id'].map(item_ctr).fillna(tr['clicked'].mean())
print(f"Popularity AUC:  {roc_auc_score(va['clicked'], va['pop']):.4f}")

# --- 2. In-history flag ---
tr_clicks = tr[tr['clicked'] == 1].groupby('user_id')['item_id'].agg(set)
va['in_hist'] = va.apply(
    lambda r: r['item_id'] in tr_clicks.get(r['user_id'], set()), axis=1)
print(f"In-history AUC:  {roc_auc_score(va['clicked'], va['in_hist']):.4f}")

# --- 3. Combined ---
va['combo'] = va['pop'] + 10 * va['in_hist']
print(f"Combined AUC:    {roc_auc_score(va['clicked'], va['combo']):.4f}")

# --- 4. Do clicked items actually appear in the event's own history? ---
va['own_hist'] = va.apply(
    lambda r: r['item_id'] in parse(r['history_item_ids']), axis=1)
print(f"\nclicked=1: {va[va.clicked==1]['own_hist'].mean():.4f} are in event history")
print(f"clicked=0: {va[va.clicked==0]['own_hist'].mean():.4f} are in event history")

# --- 5. time_bucket / day_type variability ---
print(f"\ntime_bucket unique: {train['time_bucket'].nunique()}")
print(f"day_type unique:    {train['day_type'].nunique()}")
print(f"positive rate:      {train['clicked'].mean():.4f}")