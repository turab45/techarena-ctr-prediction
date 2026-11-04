# GBDT improvement experiments

Scores supplied by the team: current GBDT benchmark 0.812910, teammate v54
0.824872, leader 0.834673. The score needed for top 15 and the submission
limit have not been supplied. These are reported scores, not independently
verified leaderboard results.

## What the local audit found

Run `python audit_gbdt.py` after the current baseline. See
`docs/gbdt-audit.json` for reproducible measurements.

| Population | Candidate AUC |
|---|---:|
| All temporal validation events | 0.833444 |
| First event per user after the validation cutoff | 0.820919 |
| First event per user on day 6 | 0.821011 |

Evaluation has 52,939 events for 52,939 users, spread over seven future days.
The existing all-event holdout overweights users with repeated events. The
first-event cohort is a closer proxy, but still does not reproduce the whole
evaluation week. Evaluation histories contain no interactions after the end
of the labeled training period. Median history age is 1.773 days in evaluation
and 1.031 days in the day-6 first-event cohort.

Cheap postprocessing tests selected slate normalization with power 0.25 on
17,569 tuning users. On 17,569 separate check users its AUC was 0.823863,
versus 0.823751 for the unchanged scores. This small, exploratory difference
does not justify prioritizing that change for submission.

## Proposed order

1. **Build two frozen-cutoff backtests.** Cut off at the start of day 4 and
   day 5. Fit only target blocks before each cutoff. For validation, select
   each user's first event after that cutoff across all remaining labeled
   days, and build every aggregate/factor model strictly before the cutoff.
   Verify that selected histories also predate the cutoff. Report daily AUC,
   first-user AUC, cold-user AUC, and event counts. These shorter horizons are
   proxies because only seven labeled days are available.
2. **Add time-aware history features.** Include latest-history age, history
   span, counts in the last 24/72 hours, exponentially decayed candidate/history
   similarities, and interactions with pre-cutoff user activity. Current
   preference features mainly use sequence position rather than elapsed time.
   Compare the same models with and without these features on both backtests.
3. **Match training population and regularize.** Compare all events against
   first-event-per-user-per-day fitting and inverse-square-root user-frequency
   weights. Test a small set of LightGBM leaf counts and minimum leaf sizes;
   select tree count using first-user validation AUC. Keep feature cutoffs
   identical for every comparison.
4. **Add a complementary model.** Test CatBoost using the same frozen feature
   matrices before building a larger neural model. Select any ensemble weights
   using aligned validation predictions, then confirm on the second backtest.
   A stronger single-model score does not establish a useful ensemble gain.
5. **Integrate v54.** Obtain its validation predictions with labels/event IDs,
   or its training code. The uploaded evaluation CSV alone cannot establish
   a local AUC or the best blend weight. Preserve the 0.824872 champion and
   track each submission's inputs, weights, and benchmark result.

## Experimental submissions prepared

Both v54 and our GBDT contain the same 529,390 unique row IDs. Their Spearman
correlation is 0.951196. V54 scores appear to be rank-based, so both inputs are
converted to percentile ranks before blending. Correlation measures prediction
similarity; it does not establish which model is correct on disagreements.

- `submission_blend_v54_90_gbdt_10.csv`: 90% v54 rank + 10% GBDT rank.
- `submission_blend_v54_80_gbdt_20.csv`: 80% v54 rank + 20% GBDT rank.

Neither blend has a measured local AUC or benchmark score. If submission slots
are scarce, validate weights using v54 holdout predictions before uploading.
With spare experimental slots, 90/10 is the more conservative first test.
The input files and the existing best submission are preserved.
