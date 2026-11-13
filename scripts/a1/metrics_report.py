'''Macro precision/recall/F1 and confusion matrices from the per-image test predictions.

analyze.py writes reports/results/a1/predictions/<run>.csv for every run (image id,
label, prediction, confidence). Everything here is computed from those files, so
it needs neither a GPU nor a checkpoint:

    python scripts/a1/metrics_report.py          # reads results_a1.csv and predictions/

It writes metrics_a1.csv and metrics_a1.tex next to the predictions, and a
confusion-matrix figure for the CNN and the Transformer with the highest
validation accuracy (chosen on val, not on test).

Macro means are over the 101 classes, as common/metrics defines them: a class
that is never predicted has precision 0.
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

from scripts.a1.texutil import bold_best

MODEL_NAMES = {'resnet50': 'ResNet-50', 'deit_s': 'DeiT-S'}

NUM_CLASSES = 101
RESULTS_DIR = _repo_root / 'reports' / 'results' / 'a1'
FIGURES_DIR = _repo_root / 'docs' / 'assets' / 'a1'
MODES = ('scratch', 'head', 'partial', 'full')


def read_predictions(path):
    '''(labels, predictions) of a predictions csv, as integer arrays.'''
    labels, preds = [], []
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            labels.append(int(row['label']))
            preds.append(int(row['pred']))
    return np.array(labels), np.array(preds)


def confusion_matrix(labels, preds, num_classes=NUM_CLASSES):
    '''Rows are true classes, columns predicted classes.'''
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (labels, preds), 1)
    return matrix


def macro_metrics(matrix):
    '''Accuracy and macro precision, recall, F1 of a confusion matrix.'''
    matrix = matrix.astype(np.float64)
    tp = np.diag(matrix)
    predicted, actual = matrix.sum(axis=0), matrix.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(tp), where=total > 0)
    return {
        'accuracy': tp.sum() / matrix.sum(),
        'precision': precision.mean(),
        'recall': recall.mean(),
        'f1': f1.mean(),
    }


def plot_confusion_matrix(matrix, classes, run, figures_dir):
    '''Row-normalized heatmap: cell (t, p) is the share of true class t predicted as p.'''
    shares = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(7, 6.2))
    image = axis.imshow(shares, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
    axis.set_xlabel('Predicted class index')
    axis.set_ylabel('True class index')
    axis.set_title(f'{run}: confusion matrix on test (row-normalized)', fontsize=9)
    ticks = list(range(0, len(classes), 10))
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label='Share of the true class')
    figure.tight_layout()
    path = Path(figures_dir) / f'confusion_{run}.png'
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def write_metrics(rows, results_dir):
    '''metrics_a1.csv and metrics_a1.tex, one row per run.'''
    fields = ('run', 'model', 'mode', 'accuracy', 'precision', 'recall', 'f1')
    with (results_dir / 'metrics_a1.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f'{row[key]:.4f}' if isinstance(row[key], float) else row[key] for key in fields})

    lines = [
        r'\begin{tabular}{llrrrr}',
        r'\toprule',
        r'Mô hình & Chế độ & Accuracy (\%) & Precision (\%) & Recall (\%) & F1 (\%) \\',
        r'\midrule',
    ]
    for row in sorted(rows, key=lambda item: (item['model'], MODES.index(item['mode']))):
        lines.append(' & '.join((
            MODEL_NAMES[row['model']],
            row['mode'],
            *(f'{100 * row[key]:.1f}' for key in ('accuracy', 'precision', 'recall', 'f1')),
        )) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    lines = bold_best(lines, higher=(2, 3, 4, 5))
    (results_dir / 'metrics_a1.tex').write_text('\n'.join(lines))


def summarize(records, classes, results_dir, figures_dir):
    '''Write the metric tables and the confusion matrices.

    records: dicts with run, model, mode and val_acc (None allowed). Returns the metric rows.
    '''
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    rows, matrices = [], {}
    for record in records:
        labels, preds = read_predictions(results_dir / 'predictions' / f"{record['run']}.csv")
        matrix = confusion_matrix(labels, preds, len(classes))
        matrices[record['run']] = matrix
        rows.append({'run': record['run'], 'model': record['model'], 'mode': record['mode'],
                     **macro_metrics(matrix)})

    write_metrics(rows, results_dir)

    figures_dir.mkdir(parents=True, exist_ok=True)
    for model in sorted({record['model'] for record in records}):
        candidates = [record for record in records if record['model'] == model and record['val_acc'] is not None]
        if candidates:
            best = max(candidates, key=lambda record: record['val_acc'])
            plot_confusion_matrix(matrices[best['run']], classes, best['run'], figures_dir)

    return rows


def _read_classes(root):
    names = (Path(root) / 'food-101' / 'meta' / 'classes.txt')
    return names.read_text().split() if names.is_file() else [str(i) for i in range(NUM_CLASSES)]


def main():
    parser = argparse.ArgumentParser(description='Macro metrics and confusion matrices from predictions.')
    parser.add_argument('--results-dir', default=str(RESULTS_DIR))
    parser.add_argument('--figures-dir', default=str(FIGURES_DIR))
    parser.add_argument('--root', default=str(_repo_root / 'data' / 'food101_256'),
                        help='prepared Food-101 folder, only for the class names')
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    records = []
    with (results_dir / 'results_a1.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            records.append({'run': row['run'], 'model': row['model'], 'mode': row['mode'],
                            'val_acc': float(row['val_acc']) if row['val_acc'] else None,
                            'test_f1': float(row['test_f1']) if row['test_f1'] else None})

    rows = summarize(records, _read_classes(args.root), results_dir, args.figures_dir)
    for row, record in zip(rows, records):
        print(f"{row['run']:18} acc {100 * row['accuracy']:.2f}  P {100 * row['precision']:.2f}  "
              f"R {100 * row['recall']:.2f}  F1 {100 * row['f1']:.2f}  (results_a1 test_f1 {100 * record['test_f1']:.2f})")


if __name__ == '__main__':
    main()
