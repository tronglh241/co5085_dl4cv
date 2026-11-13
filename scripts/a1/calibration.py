'''Calibration of the A1 runs from their per-image test predictions.

analyze.py writes reports/results/a1/predictions/<run>.csv (image id, label,
prediction, softmax confidence of the prediction). From those files alone, with
no GPU or checkpoint, this script measures how far each run's confidence is from
its accuracy:

- ECE: expected calibration error over equal-width confidence bins, the mean
  |accuracy - confidence| of the bins weighted by their share of images,
- the mean confidence minus the accuracy (below 0: underconfident),
- AUROC of the confidence as a score that separates right from wrong
  predictions (0.5: no better than chance),
- the accuracy on the 80% of images with the highest confidence (what a model
  that may abstain on its least sure 20% would score).

Every run trains with label smoothing 0.2 over 101 classes, so the loss is
lowest when the true class gets 1 - 0.2 + 0.2/101 = 0.802 of the probability.
A run that fits its training data therefore rarely says more than about 0.8,
even on images it gets right.

    python scripts/a1/calibration.py

writes calibration_a1.csv/.tex to reports/results/a1 and reliability_a1.png to
docs/assets/a1.
'''

import argparse
import csv
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = _repo_root / 'reports' / 'results' / 'a1'
FIGURES_DIR = _repo_root / 'docs' / 'assets' / 'a1'
MODELS = ('resnet50', 'deit_s')
MODES = ('scratch', 'head', 'partial', 'full')
LABEL_SMOOTHING = 0.2
NUM_CLASSES = 101
BINS = 15
COVERAGE = 0.8
# Bins with fewer images are left out of the reliability plot (too noisy to read).
MIN_BIN_IMAGES = 50


def read_run(path):
    '''(correct, confidence) arrays of one predictions csv.'''
    correct, confidence = [], []
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            correct.append(row['label'] == row['pred'])
            confidence.append(float(row['confidence']))
    return np.array(correct, dtype=bool), np.array(confidence)


def bin_stats(correct, confidence, bins=BINS):
    '''Per-bin (count, mean confidence, accuracy) over equal-width bins of [0, 1].'''
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    count = np.bincount(index, minlength=bins)
    conf_sum = np.bincount(index, weights=confidence, minlength=bins)
    hit_sum = np.bincount(index, weights=correct.astype(float), minlength=bins)
    safe = np.maximum(count, 1)
    return count, conf_sum / safe, hit_sum / safe


def expected_calibration_error(correct, confidence, bins=BINS):
    count, mean_conf, accuracy = bin_stats(correct, confidence, bins)
    return float(np.sum(count / count.sum() * np.abs(accuracy - mean_conf)))


def auroc(scores, positive):
    '''Area under the ROC curve of `scores` for the boolean `positive`, with tied scores averaged.'''
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average rank over each group of equal scores.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inverse, weights=ranks) / counts)[inverse]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def selective_accuracy(correct, confidence, coverage=COVERAGE):
    '''Accuracy on the `coverage` share of images with the highest confidence.'''
    keep = max(1, int(round(coverage * len(correct))))
    order = np.argsort(-confidence, kind='mergesort')
    return float(correct[order[:keep]].mean())


def summarize_run(correct, confidence):
    return {
        'accuracy': float(correct.mean()),
        'mean_conf': float(confidence.mean()),
        'conf_minus_acc': float(confidence.mean() - correct.mean()),
        'ece': expected_calibration_error(correct, confidence),
        'auroc': auroc(confidence, correct),
        'acc_at_80': selective_accuracy(correct, confidence),
        'share_above_target': float(np.mean(confidence > 1 - LABEL_SMOOTHING + LABEL_SMOOTHING / NUM_CLASSES)),
    }


def write_tables(rows, results_dir):
    fields = ('run', 'model', 'mode', 'accuracy', 'mean_conf', 'conf_minus_acc', 'ece', 'auroc',
              'acc_at_80', 'share_above_target')
    with (results_dir / 'calibration_a1.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f'{row[key]:.4f}' if isinstance(row[key], float) else row[key] for key in fields})

    lines = [
        r'\begin{tabular}{llrrrrrr}',
        r'\toprule',
        r'Model & Mode & Acc (\%) & Conf (\%) & Conf $-$ Acc & ECE (\%) & AUROC & Acc@80\% (\%) \\',
        r'\midrule',
    ]
    for row in rows:
        lines.append(' & '.join((
            row['model'].replace('_', r'\_'),
            row['mode'],
            f"{100 * row['accuracy']:.1f}",
            f"{100 * row['mean_conf']:.1f}",
            f"{100 * row['conf_minus_acc']:+.1f}",
            f"{100 * row['ece']:.1f}",
            f"{row['auroc']:.3f}",
            f"{100 * row['acc_at_80']:.1f}",
        )) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    (results_dir / 'calibration_a1.tex').write_text('\n'.join(lines))


def plot_reliability(curves, figures_dir):
    '''One panel per model, one line per mode: accuracy against mean confidence of each bin.'''
    target = 1 - LABEL_SMOOTHING + LABEL_SMOOTHING / NUM_CLASSES
    figure, axes = plt.subplots(1, len(MODELS), figsize=(10, 4.2), squeeze=False)
    for axis, model in zip(axes[0], MODELS):
        axis.plot([0, 1], [0, 1], color='0.6', linestyle='--', linewidth=1, label='perfect calibration')
        axis.axvline(target, color='0.75', linestyle=':', linewidth=1)
        axis.text(target, 0.03, ' 0.802', fontsize=7, color='0.4')
        for mode in MODES:
            if (model, mode) not in curves:
                continue
            count, mean_conf, accuracy = curves[(model, mode)]
            keep = count >= MIN_BIN_IMAGES
            axis.plot(mean_conf[keep], accuracy[keep], marker='o', markersize=3, label=mode)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect('equal')
        axis.set_title(model)
        axis.set_xlabel('Mean confidence of the bin')
        axis.set_ylabel('Accuracy of the bin')
        axis.legend(fontsize=7, loc='upper left')
    figure.tight_layout()
    figure.savefig(Path(figures_dir) / 'reliability_a1.png', dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Calibration of the A1 runs from their test predictions.')
    parser.add_argument('--results-dir', default=str(RESULTS_DIR))
    parser.add_argument('--figures-dir', default=str(FIGURES_DIR))
    args = parser.parse_args()
    results_dir, figures_dir = Path(args.results_dir), Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows, curves = [], {}
    for model in MODELS:
        for mode in MODES:
            run = f'{model}_{mode}'
            path = results_dir / 'predictions' / f'{run}.csv'
            if not path.is_file():
                print(f'skip {run}: no {path}')
                continue
            correct, confidence = read_run(path)
            rows.append({'run': run, 'model': model, 'mode': mode, **summarize_run(correct, confidence)})
            curves[(model, mode)] = bin_stats(correct, confidence)

    if not rows:
        raise SystemExit(f'no predictions under {results_dir / "predictions"}: run analyze.py first')

    write_tables(rows, results_dir)
    plot_reliability(curves, figures_dir)
    for row in rows:
        print(f"{row['run']:18} acc {100 * row['accuracy']:5.1f}  conf {100 * row['mean_conf']:5.1f}  "
              f"ECE {100 * row['ece']:4.1f}  AUROC {row['auroc']:.3f}  acc@80 {100 * row['acc_at_80']:5.1f}  "
              f">0.802 {100 * row['share_above_target']:4.1f}%")


if __name__ == '__main__':
    main()
