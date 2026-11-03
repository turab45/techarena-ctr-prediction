"""Regenerate the documented training curve: uv run --with matplotlib python docs/plot_training.py."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    docs = Path(__file__).resolve().parent
    run = json.loads((docs/'results/training-2026-11-03.json').read_text())
    rows = run['epochs']
    assert [r['epoch'] for r in rows] == list(range(1, 31))
    best = max(rows, key=lambda row: row['val_auc'])
    assert (best['epoch'], best['val_auc']) == (run['best_epoch'], run['best_validation_auc'])

    plt.rcParams.update({'font.size': 11, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'techarena-training-2026-11-03'})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    epochs = [row['epoch'] for row in rows]
    axes[0].plot(epochs, [row['train_loss'] for row in rows], color='#2563eb', linewidth=2)
    axes[0].set(title='Training loss', ylabel='Combined loss')
    axes[1].plot(epochs, [row['val_auc'] for row in rows], color='#15803d', linewidth=2)
    axes[1].scatter([best['epoch']], [best['val_auc']], color='#b45309', s=45, zorder=3)
    axes[1].annotate(f"Best: {best['val_auc']:.5f} at epoch {best['epoch']}",
                     xy=(best['epoch'], best['val_auc']), xytext=(8, 0.70),
                     arrowprops={'arrowstyle': '->', 'color': '#b45309'})
    axes[1].set(title='Local validation ROC AUC', ylabel='ROC AUC', ylim=(0.55, 0.84))
    for ax in axes:
        ax.axvspan(1, 2.5, color='#94a3b8', alpha=0.16)
        ax.axvline(best['epoch'], color='#b45309', linestyle='--', linewidth=1)
        ax.set(xlabel='Epoch', xlim=(1, 30), xticks=[1, 5, 10, 15, 20, 25, 30])
        ax.grid(alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Item2vec + slate transformer — supplied 30-epoch run')
    output = docs/'figures/training-curve.svg'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={'Date': None})
    output.write_text('\n'.join(line.rstrip() for line in output.read_text().splitlines()) + '\n')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
