# Methods and implementation

The current source is the GPU transformer pipeline in [`pipeline.py`](../pipeline.py), with [`diag.py`](../diag.py) for diagnostics and [`pretrain_item2vec.py`](../pretrain_item2vec.py) for unsupervised item pretraining. The project is associated with [Huawei Ireland Tech Arena 2026](https://techarena-ireland.hackathon.com/); dataset handling and model details below describe this implementation.

## 1. Event-level preparation and validation

The input CSVs contain one candidate per row. Preparation groups candidates by `event_id`, preserving their original within-event order. It collects their item IDs and labels, or evaluation row IDs, into a single event record. User history IDs and history timestamps are parsed from JSON. These records are cached as pickles.

The vocabulary includes candidate items, history items, and users from both train and evaluation data. Item/user index zero is reserved for padding or unknown IDs. The model defaults to ten candidate slots and the latest twenty history items. For each history item it computes the nonnegative age in hours relative to the event. It also counts how often each candidate occurs in that retained history.

Validation holds out events at or above the 85th percentile of training event timestamps. Earlier events are used for fitting. For the supplied run:

| Split property | Value |
|---|---|
| Training events | 583,790 |
| Validation events | 103,034 |
| Timestamp cutoff | 2025-01-25 23:14:00 UTC |
| Training-file span | 2025-01-20 through 2025-01-26 |
| Evaluation-file span | 2025-01-27 through 2025-02-02 |

ROC AUC is computed over flattened candidate labels and sigmoid scores across all validation events. This differs from evaluating only each user's first event on a particular date, as some reference experiments did.

## 2. Diagnostic baselines

`diag.py` checks whether simple signals already separate clicks from non-clicks:

| Check | How it works | What it helps assess |
|---|---|---|
| Item popularity | Mean training click rate per item, with a global-mean fallback | How much ranking signal comes from global item preference |
| Prior-click overlap | Whether a candidate appears among the user's clicks in the fitting period | The usefulness of user-specific repeat interactions |
| Combined score | Popularity plus ten times the prior-click indicator | Whether those two simple signals complement each other |
| Event-history overlap | Candidate membership in its own supplied event history, compared for positives and negatives | Whether the histories are relevant to the target event |
| Context variability | Distinct `time_bucket` and `day_type` counts, plus the positive rate | Whether context features vary and how sparse clicks are |

The diagnostic script splits CSV rows at a timestamp quantile; classifier training splits event records. They need not select identical events if slate sizes vary. Its prior-click indicator also differs from the classifier's supplied-history count feature.

The diagnostic run on 3 November 2026 measured popularity AUC **0.5275**, prior-click-overlap AUC **0.5346**, and combined AUC **0.5523**. A candidate appears in its own supplied history for **12.10% of positives** and **1.98% of negatives**; the overall positive rate is **6.29%**. These are simple baseline measurements, not an upper bound on what a learned interaction model can achieve. The rounded outputs are preserved in [JSON](results/diagnostics-2026-11-03.json).

The cached training data has four time buckets and two day types. Both context embeddings were retained.

## 3. Initial transformer and stability fixes

The initial configuration used random 128-dimensional item/user embeddings, two history encoder layers, two slate encoder layers, and feed-forward width 512. An earlier report described validation AUC around 0.56; a complete baseline log is not included in this repository. The working hypothesis was that sparse click supervision was insufficient to learn useful item representations from scratch.

Empty histories exposed a separate numerical bug: masking every history key can produce undefined attention weights and NaNs. The dataset now inserts one padding-item placeholder at the event timestamp. The history mask additionally clamps lengths to at least one, guaranteeing an unmasked slot. The item embedding at index zero remains zero, while history-position and gap embeddings provide a learned representation for that slot.

The printed `hist_len==0` count is measured after this transformation. Zero counts do not establish that all raw histories were nonempty. The recorded run reports zero transformed empty lengths in both splits.

Training now stops on a non-finite loss before backpropagation or an optimizer update. The SmoothAUC weight was lowered from 0.2 to 0.1. `test.sh` stops on the first failed command, avoiding prediction and blending errors caused by missing upstream outputs.

## 4. Item2vec pretraining

Word2Vec skip-gram treats item IDs as tokens and user history snapshots as sequences. Items that occur in similar history contexts can obtain related embeddings before supervised click training.

| Setting | Value |
|---|---|
| Input | History sequences from training and evaluation CSVs |
| Sequence selection | At least five parsed items |
| Deduplication | Raw history strings deduplicated within each input file |
| Vector size | 64 |
| Context window | 10 |
| Objective | Skip-gram with five negative samples |
| Minimum token count | 1 |
| Epochs / CPU workers | 20 / 16 |

The saved dictionary maps original item IDs to vectors. The classifier aligns those IDs through `item2idx`; Word2Vec's internal ordering is not used as the classifier's row ordering. Loading preserves padding row zero and leaves unmatched item rows randomly initialized. If dimensions differ, it copies the shared leading dimensions. Invalid/non-finite vectors and a completely unmatched vocabulary cause an error.

The recorded run loaded **49,974 of 50,000** item vectors, approximately **99.95% coverage**. The other 26 items retained random initialization.

## 5. Current scoring model

The model contains approximately 6.82 million parameters for the recorded vocabulary of 50,000 items and 54,456 users.

1. **History encoder.** Add item, history-position, and bucketed-age embeddings. A transformer encoder contextualizes the history, with padding masked.
2. **Candidate slate encoder.** Add candidate item, slate-position, user, time-bucket, and day-type embeddings. A separate transformer models relationships among candidates in the same event.
3. **Cross-attention.** Each candidate attends to the encoded user history. A residual connection and layer normalization combine that information with the candidate representation.
4. **History pooling and scoring head.** Masked mean pooling summarizes the valid history, masked max pooling takes the strongest coordinate values, and last-item pooling selects the last valid position rather than the padded tail. All three vectors are broadcast to each candidate and concatenated with its contextual representation, attended history, and nine numeric values: history count divided by 20, smoothed item CTR, log-scaled item exposure count, log-scaled user activity, smoothed user CTR, and sine/cosine encodings of UTC hour and weekday. A small MLP returns one logit per candidate; sigmoid converts it to a score for validation and prediction.

The defaults are 64-dimensional embeddings, four attention heads, two history encoder layers, one slate encoder layer, feed-forward width 256, and dropout 0.10, matching the original recorded run's dropout.

P2 fits smoothed item/user statistics on the fitting portion only and reuses them unchanged for validation, so held-out labels do not enter those statistics. Training-row statistics do include their own fitting labels; this is aggregate target encoding, not an out-of-fold or strictly past-only encoding. Candidate item age has been removed; history time gaps are retained. Prediction recomputes the statistics from the full labeled training cache as requested. Missing item/user statistics leave their numeric features at zero. With P3 pooling, the head has `5*d_model + 9` inputs, or `5*d_model + 1` without numeric features. New checkpoints record the pooling flag; prediction disables pooling for earlier checkpoints that lack it. Ten-feature P2 checkpoints still require retraining for the nine-feature numeric schema.

The objective is:

```text
loss = BCE + 0.3 × ListNet + 0.1 × SmoothAUC
```

- **BCE** supervises the click label of every candidate.
- **ListNet** uses a softmax over a slate and a cross-entropy target at its clicked position. Slates without a positive are omitted from this term. The current target uses `argmax`, assuming at most one clicked item per slate.
- **SmoothAUC** averages `sigmoid((negative_logit - positive_logit) / 0.5)` across positive-negative pairs within each slate, then equally averages over slates with such pairs. It encourages clicked items to outrank unclicked competitors.

This implementation's ListNet term uses a single clicked-position target; it is not a general multi-click listwise distribution objective.

## 6. Training schedule and GPU execution

The current defaults use AdamW with learning rate `1e-3`, weight decay `1e-5`, batch size 512, gradient norm clipping at 5, and a maximum of 15 epochs. Training stops after four consecutive validation epochs without a strict AUC improvement. A first checkpoint is saved even if its AUC is zero. Item embeddings are frozen in epochs 1–2 and become trainable at epoch 3. All parameters remain in the optimizer, so the unfreezing does not require adding a new parameter group.

History dropout was removed in P3 Patch 1. Training, validation, and prediction all use the full retained history, time gaps, and length. The dataset's `self.training` flag remains but no longer affects samples. History items and their gaps are returned directly from the cached arrays.

The original 30-epoch experiment instead used learning rate `1e-3`, weight decay `1e-5`, dropout 0.1, and no early stopping or history dropout. Its historical JSON retains those original settings.

The learning-rate multiplier grows linearly from zero over the first 500 optimizer updates, then follows cosine decay over the planned run. The scheduler advances after each successful optimizer update, not after each epoch. If automatic mixed precision skips an overflowing update, the scheduler does not advance for that batch.

CUDA is the default training device. The model and every batch tensor are transferred there, and the first batch prints the model, candidate tensor, and logits devices. CPU training requires an explicit `--device cpu`. GPU training uses FP16 autocast and gradient scaling; CSV preparation, item2vec, and dataset construction remain CPU work.

Two performance changes reduce GPU overhead:

- Batched SmoothAUC computes all within-slate pairs together, replacing a Python loop with GPU indexing and kernel launches for every slate.
- Cross-attention uses `need_weights=False`, since the scoring head does not consume attention-weight matrices.

A development microbenchmark on the RTX 4090 measured SmoothAUC forward/backward at **96.53 ms before** and **0.32 ms after** batching for 512 slates of ten items. Loss values and gradients matched the original calculation in that check. This measures the loss computation only; it is not a claim of a comparable whole-training speedup.

## 7. Prediction and optional rank blending

Prediction reconstructs the architecture from the checkpoint configuration, restores its weights, and maps the resulting scores to evaluation `rowid` values. It does not reload item2vec: the learned embeddings are already inside the checkpoint.

The optional blender merges two `rowid,predicted_probability` files, converts each prediction column to percentile ranks, and takes a weighted average. It is intended to combine complementary rankings, such as a tree model and the sequence/slate model. No new-transformer blend result has been measured here. It rejects duplicate or missing IDs, mismatched ID coverage, nonfinite scores, and weights outside [0, 1]. Rank averages should not be interpreted as calibrated probabilities.

## Reference methods from the original repository

This project began from a clone of [Engr-Riaz-Ahmad-Butt/techarena-ctr-prediction](https://github.com/Engr-Riaz-Ahmad-Butt/techarena-ctr-prediction) and was rewritten around the current transformer. The source project's tree and collaborative-filtering experiments provide context; their deleted implementation files and model artifacts are not part of this repository.

| Reference method | Explanation |
|---|---|
| CTR statistics and tree models | Smoothed click-rate features and user/item statistics give CatBoost or LightGBM useful tabular inputs |
| SVD and exposure-aware ALS | Latent user/item factors summarize collaborative structure and exposure-adjusted affinities |
| Aligned ALS interactions | Alignment makes factors from different temporal fits comparable before building component-wise interaction features |
| Event-choice and ensemble models | Model within-event competition and combine predictions through probability, log-odds, or rank blends |

The source project's saved README reports an aligned-ALS tree blend at local AUC **0.821528 on 26 January** and **0.814600 on 25 January**, using first-event cohorts. These are attributed reference results, not results produced by this rewrite. Their validation populations differ from the current 15% event-time holdout, so the numbers are not a matched comparison. No official score for the new transformer has been supplied.
