# Candidate position and three-seed ensemble

Implemented the supplied position-feature and ensemble proposal with two fixes:
the original similarity-relative columns retain their meanings after inserting
position at column 13, and both validation and final prediction average all
three models. Each final seed uses its own validation-selected tree count.

The feature width is 355 (191 preference, 68 warm, 32 context, 64 ALS).
New caches live in `cache/features_gbdt_v5/`; old caches and root baseline
outputs are preserved. Assembled caches validate both row and column counts.
On 1,000 sampled rows from the first rebuilt block, removing position exactly
reproduced all 354 previous feature values.

| Setting | Previous model | New ensemble |
|---|---:|---:|
| Seeds | 42 | 42, 43, 44 |
| Leaves | 31 | 63 |
| Learning rate | 0.04 | 0.03 |
| Minimum rows per leaf | 200 | 100 |
| Feature fraction | 0.9 | 0.7 |
| Features | 354 | 355 |

Validation uses the same temporal cutoff, candidate rows, and cohort as the
previous model. All new scores below are local measurements.

| Model | Overall AUC | Selected trees |
|---|---:|---:|
| Previous model | 0.833444 | 891 |
| New seed 42 | 0.833526 | 840 |
| New seed 43 | 0.833519 | 1,058 |
| New seed 44 | 0.833465 | 913 |
| New probability average | **0.833873** | 840 / 1,058 / 913 |

First-event-per-user day-6 cohort AUC increased from **0.821011 to 0.821618**
on 35,138 events. The overall gain is 0.000429; the cohort gain is 0.000608.
These experiments changed capacity, subsampling, position, and seed averaging
together; they do not isolate the causal contribution of each change.

The claim of strong position bias was not supported by the data check:
position-level training CTRs ranged from 0.062545 to 0.063208, and using those
training CTRs as position-only validation scores yielded AUC 0.500244.
Position ranked 325th and 316th out of 355 features by gain in seeds 42 and 43.
CSV candidate order is the position feature; the source has no explicit
display-position column.

Run artifacts are saved under `artifacts/gbdt_position_v5/`: validation and
final model files per seed, averaged validation predictions, `metrics.json`,
`audit.json`, and (after the full run completes) `submission_gbdt.csv`.
The new submission has no measured leaderboard score yet.

Six targeted tests pass in `test_gbdt_ensemble.py` and `test_gbdt_runtime.py`.
They cover candidate ordering, relative-feature indices, component boundaries,
cache-width invalidation, seed averaging, final tree counts, chunked CSV
preparation, and Mac thread defaults.

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run python -u gbdt.py --rebuild-features
uv run python audit_gbdt.py --run-dir artifacts/gbdt_position_v5
```
