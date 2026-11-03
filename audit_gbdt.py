"""Audit the current saved model against the one-event-per-user evaluation set.

Run after gbdt.py. This never overwrites models or submission files.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DAY = 86400


def first_events(users, events):
    """Select the earliest supplied event for each user; events must be sorted."""
    _, positions = np.unique(users[events], return_index=True)
    return np.sort(events[positions])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path,
                        help='audit a new ensemble run; default: the original root baseline')
    args = parser.parse_args()
    run_dir = args.run_dir or Path('.')
    with np.load('cache/gbdt_data.npz') as data:
        ts, users = data['tr_ts'], data['tr_user']
        all_labels = data['tr_labels']
        eval_ts, eval_users = data['ev_ts'], data['ev_user']
        histories = {prefix: (data[prefix + '_hist'], data[prefix + '_hts'])
                     for prefix in ('tr', 'ev')}
    start = int(ts.min())
    cutoff = int(np.quantile(ts, .85))
    events = np.flatnonzero(ts >= cutoff)
    frame = pd.read_parquet(run_dir / 'val_preds_gbdt.parquet')
    pred = frame.pred.to_numpy().reshape(-1, 10)
    labels = frame.label.to_numpy().reshape(-1, 10)
    assert len(pred) == len(events)
    assert np.array_equal(labels, all_labels[events])
    assert np.array_equal(frame.event_row.to_numpy(), np.repeat(np.arange(len(events)), 10))
    if args.run_dir:
        metrics = json.loads((run_dir / 'metrics.json').read_text())
        feature_cache = Path(metrics['feature_cache'])
    else:
        feature_cache = Path('cache/features_gbdt_v4')
    features = np.load(feature_cache / 'val_matrix.npy', mmap_mode='r')
    assert features.shape[0] == len(events) * 10 and features.shape[1] in (354, 355)

    def auc(rows, scores=pred):
        return float(roc_auc_score(labels[rows].ravel(), scores[rows].ravel()))

    cohorts = {'all_validation': events,
               'first_after_cutoff': first_events(users, events),
               'day6_first': first_events(users, events[ts[events] >= start + 6 * DAY])}
    report = {'cutoff': cutoff, 'cohorts': {}, 'evaluation': {
        'events': len(eval_ts), 'users': len(np.unique(eval_users)),
        'day_counts': {str(int(day)): int(count) for day, count in zip(
            *np.unique((eval_ts - start) // DAY, return_counts=True))}}}
    for name, selected in cohorts.items():
        rows = np.searchsorted(events, selected)
        report['cohorts'][name] = {'events': len(rows), 'auc': auc(rows),
                                     'positive_rate': float(labels[rows].mean())}
        items, stamps = histories['tr']
        latest = np.where(items[selected] >= 0, stamps[selected], -1).max(axis=1)
        nonempty = latest >= 0
        report['cohorts'][name]['median_history_age_days'] = float(np.median(
            (ts[selected][nonempty] - latest[nonempty]) / DAY))
    for prefix, (items, stamps) in histories.items():
        target_ts = ts if prefix == 'tr' else eval_ts
        latest = np.where(items >= 0, stamps, -1).max(axis=1)
        lag = (target_ts[latest >= 0] - latest[latest >= 0]) / DAY
        report[prefix + '_history_age_days'] = {
            str(q): float(np.quantile(lag, q)) for q in (.25, .5, .75, .9, .99)}
        report[prefix + '_history_after_training_end_fraction'] = float(
            ((stamps >= start + 7 * DAY) & (items >= 0)).any(axis=1).mean())

    # Tune cheap score combinations on half the users, then evaluate the single
    # selected setting on the other half. This remains exploratory validation.
    rows = np.searchsorted(events, cohorts['day6_first'])
    rng = np.random.default_rng(2026)
    shuffled = rng.permutation(rows)
    tune, check = shuffled[:len(rows)//2], shuffled[len(rows)//2:]
    warm_column = 191 if features.shape[1] == 355 else 190
    warm = np.asarray(features[:, warm_column]).reshape(-1, 10)
    def logit(p):
        p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
        return np.log(p) - np.log1p(-p)
    base_logits, warm_logits = logit(pred), logit(warm)
    candidates = {'baseline': pred}
    for weight in (.1, .25, .5, .75, 1.):
        candidates[f'warm_logit_weight_{weight}'] = (1-weight)*base_logits + weight*warm_logits
    for power in (.25, .5, 1.):
        candidates[f'slate_normalization_{power}'] = pred / pred.sum(axis=1, keepdims=True)**power
    tuned = {name: auc(tune, scores) for name, scores in candidates.items()}
    best = max(tuned, key=tuned.get)
    report['score_experiment'] = {
        'selection': 'random_seed_2026_disjoint_user_halves_of_day6_first',
        'tuning_events': len(tune), 'check_events': len(check),
        'tuning_auc': tuned, 'selected': best,
        'baseline_check_auc': auc(check),
        'selected_check_auc': auc(check, candidates[best]),
        'warning': 'Exploratory; confirm on an earlier frozen-cutoff fold before submission.'}
    path = run_dir / 'audit.json' if args.run_dir else Path('docs/gbdt-audit.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
