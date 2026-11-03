# pretrain_item2vec.py
import json, pickle
import pandas as pd
from gensim.models import Word2Vec

def parse(s):
    try: return json.loads(s)
    except: return []

train = pd.read_csv("data/train.csv", usecols=['history_item_ids'])
ev    = pd.read_csv("data/evaluation.csv", usecols=['history_item_ids'])

# dedupe: many events share the same user's history snapshot
seqs = []
for df in (train, ev):
    for h in df['history_item_ids'].drop_duplicates().values:
        s = parse(h)
        if len(s) >= 5:
            seqs.append(s)
print(f"sequences: {len(seqs):,}")

model = Word2Vec(
    seqs,
    vector_size=64,     # matches d_model below
    window=10,          # 20-item history → window 10 sees the whole neighbourhood
    min_count=1,
    sg=1,               # skip-gram: better for rare items (most items are rare here)
    negative=5,
    workers=16,
    epochs=20,
)

emb = {item: model.wv[item].copy() for item in model.wv.index_to_key}
with open("cache/item2vec.pkl", "wb") as f:
    pickle.dump(emb, f)
print(f"saved {len(emb):,} item embeddings  (dim={model.wv.vector_size})")

# Sanity: nearest neighbours of an item
sample = next(iter(emb))
print(f"\nneighbours of {sample}:")
for w, _ in model.wv.most_similar(sample, topn=5):
    print(f"  {w}")