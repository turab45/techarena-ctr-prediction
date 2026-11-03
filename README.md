# TechArena CTR Prediction

GPU training for click-through ranking using pretrained item embeddings, user histories, and a transformer that scores a slate of candidate items together.

**Best recorded local validation ROC AUC: 0.81292 at epoch 4.** This is from the supplied 30-epoch training log, not an official leaderboard score. The run used an NVIDIA RTX 4090 and a model with approximately 6.82 million parameters. See the [experiment record](docs/experiments.md) for the full curve, settings, and interpretation.

## Hackathon

This project is for [Huawei Ireland Tech Arena 2026](https://techarena-ireland.hackathon.com/). The event explores sequential recommendations in a competitive digital marketplace: using interaction histories to anticipate responses while respecting compute and runtime constraints. This repository implements supervised click scoring for the supplied candidate slates.

The organizer lists the first phase as **21 October–4 November 2026** and the onsite event as **14–15 November 2026 in Dublin**. See the linked event page for the organizer's schedule and challenge overview.

## Approach

```mermaid
flowchart LR
    H[User history sequences] --> W[Item2vec pretraining]
    W --> I[64-dimensional item embeddings]
    H --> E[History transformer]
    I --> E
    I --> S[Candidate slate transformer]
    E --> A[Candidate-to-history attention]
    S --> A
    A --> P[Click scoring head]
```

The scoring head also receives the count of each candidate in the user's history. Candidate representations include user, position, time-bucket, and day-type embeddings. Training combines binary cross-entropy, ListNet, and a smooth pairwise AUC objective.

- [Methods and implementation](docs/methods.md): preprocessing, diagnostics, model, losses, GPU changes, and reference methods.
- [Experiments and results](docs/experiments.md): what we tried, the complete 30-epoch record, and current limitations.
- [Machine-readable training results](docs/results/training-2026-11-03.json).

## Setup

The project uses `uv`, pins Python 3.9 in `.python-version`, and records dependencies in `uv.lock`. Training defaults to CUDA and stops if no GPU is available; CPU runs require `--device cpu`. Item2vec pretraining and CSV preparation run on CPU.

```bash
uv sync --locked
```

Obtain the competition dataset separately and place it at:

```text
data/train.csv
data/evaluation.csv
```

Both files need `event_id`, `user_id`, `item_id`, `timestamp`, `time_bucket`, `day_type`, `history_item_ids`, and `history_timestamps`. Training also needs `clicked`; evaluation needs `rowid`. Histories are JSON lists and timestamps are Unix seconds. The default model assumes ten candidate slots and uses the latest twenty history items. Candidate order comes from CSV row order within each event.

## Run

Run these commands from the repository root. Preparation creates `cache/`; item2vec must finish before classifier training.

```bash
# 1. Build event-level caches and the shared item/user vocabulary.
uv run python pipeline.py prep

# 2. Optional diagnostics: popularity, click overlap, and feature variability.
uv run python diag.py

# 3. Pretrain 64-dimensional item vectors from history sequences.
uv run python pretrain_item2vec.py

# 4. Train the classifier on CUDA.
uv run python pipeline.py train --device cuda --epochs 30 --bs 512

# 5. Score evaluation candidates using the best saved checkpoint.
uv run python pipeline.py predict
```

Once item2vec has been generated, `bash test.sh` runs preparation and the 30-epoch training command. It does not run diagnostics, item2vec, prediction, or blending.

Training freezes item embeddings in epochs 1–2 and unfreezes them at epoch 3. It uses a learning rate of `1e-3`, 500 optimizer updates of linear warmup, then cosine decay. Each validation improvement overwrites `best_model.pt`; prediction reads that checkpoint rather than the final epoch's weights. The recorded run's best epoch was 4, while its final AUC was 0.77834. Early stopping is not implemented.

Keep the vocabulary and evaluation cache from the same preparation run as the checkpoint. Caches are reused by filename; after changing input data, regenerate the corresponding caches, vocabulary, pretrained vectors, and model together.

## Outputs and optional blending

| Artifact | Purpose |
|---|---|
| `cache/train.pkl`, `cache/eval.pkl` | One row per event, with candidate lists and parsed histories |
| `cache/vocab.pkl` | Item/user IDs mapped to embedding indices |
| `cache/item2vec.pkl` | Pretrained item vectors |
| `best_model.pt` | Best validation checkpoint and model configuration |
| `submission_dl.csv` | Evaluation `rowid,score` pairs |

The pipeline can rank-average an existing GBDT submission with the transformer scores:

```bash
uv run python pipeline.py blend \
  --a submission_gbdt.csv --b submission_dl.csv \
  --out submission_blend.csv --wa 0.4
```

Here `--wa 0.4` gives the first file 40% weight. This is an example configuration; no blended result is recorded for the new transformer. Both files must cover the same unique row IDs: the current blender uses an inner join and does not validate complete coverage. Rank-blended scores are ranking scores, not calibrated click probabilities. Uploading a submission is a separate step.

## Checks

```bash
uv run python -m unittest -v test_pipeline
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('PyTorch CUDA build:', torch.version.cuda)"
```

The tests cover embedding alignment, padding, invalid vectors, checkpoint restoration, CLI defaults, and freeze/unfreeze and scheduler behavior. The GPU integration test runs when CUDA is available and otherwise skips. It uses synthetic data and does not reproduce the full competition result.

Datasets, preprocessed caches, model weights, predictions, and the old local Git backup are excluded by `.gitignore`. The small experiment JSON and training-curve figure under `docs/` are committed documentation.
