'''Do the CNN and the Transformer make the same mistakes? From the test predictions only.

Reads reports/results/a1/predictions/<run>.csv (written by analyze.py) and writes:

- agreement_a1.csv/.tex: for each training mode, ResNet-50 against DeiT-S on the
  same images: share both right, only one right, both wrong, the error overlap
  (Jaccard index of the two sets of wrong images), how often two wrong answers
  name the same class, the accuracy of an oracle that is right when either
  model is, and of a confidence ensemble that keeps the answer of the more
  confident model (the predictions keep only the top confidence, so a full
  probability average is not possible),
- error_overlap_a1.png: the Jaccard index of the wrong-image sets of every pair
  of the 8 runs, to see whether mistakes group by architecture or by
  pretraining,
- class_diff_a1.csv: per class and mode, DeiT-S accuracy minus ResNet-50
  accuracy with an exact McNemar p-value (no multiple-test correction: 101
  classes per mode, so about 5 per mode fall below 0.05 by chance alone),
- class_diff_a1.png: the classes that favor each architecture most in full
  finetuning, the mode in which both models train every weight.

    python scripts/a1/agreement.py
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

from scripts.a1.stats import mcnemar_exact

RESULTS_DIR = _repo_root / 'reports' / 'results' / 'a1'
FIGURES_DIR = _repo_root / 'docs' / 'assets' / 'a1'
MODELS = ('resnet50', 'deit_s')
MODES = ('scratch', 'head', 'partial', 'full')
TOP_CLASSES = 12
PLOT_MODE = 'full'


def read_predictions(path):
    '''ids, labels, predictions and confidences of one predictions csv.'''
    ids, labels, preds, confidence = [], [], [], []
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            ids.append(row['id'])
            labels.append(int(row['label']))
            preds.append(int(row['pred']))
            confidence.append(float(row['confidence']))
    return ids, np.array(labels), np.array(preds), np.array(confidence)


def class_names(ids, labels, num_classes):
    '''Class names from the image ids (<class>/<image id>), indexed by label.'''
    names = [str(index) for index in range(num_classes)]
    for image_id, label in zip(ids, labels):
        names[label] = image_id.split('/', 1)[0]
    return names


def jaccard(a, b):
    union = np.sum(a | b)
    return float(np.sum(a & b) / union) if union else 0.0


def pair_stats(labels, a, b):
    '''ResNet-50 run a against DeiT-S run b, each a dict with pred and confidence arrays.'''
    right_a, right_b = a['pred'] == labels, b['pred'] == labels
    both_wrong = ~right_a & ~right_b
    ensemble = np.where(a['confidence'] >= b['confidence'], a['pred'], b['pred'])
    return {
        'acc_resnet': float(right_a.mean()),
        'acc_deit': float(right_b.mean()),
        'both_right': float(np.mean(right_a & right_b)),
        'only_resnet': float(np.mean(right_a & ~right_b)),
        'only_deit': float(np.mean(~right_a & right_b)),
        'both_wrong': float(both_wrong.mean()),
        'error_jaccard': jaccard(~right_a, ~right_b),
        'same_wrong_class': float(np.mean(a['pred'][both_wrong] == b['pred'][both_wrong])) if both_wrong.any() else 0.0,
        'oracle': float(np.mean(right_a | right_b)),
        'ensemble': float(np.mean(ensemble == labels)),
    }


def write_agreement(rows, results_dir):
    fields = ('mode', 'acc_resnet', 'acc_deit', 'both_right', 'only_resnet', 'only_deit', 'both_wrong',
              'error_jaccard', 'same_wrong_class', 'oracle', 'ensemble')
    with (results_dir / 'agreement_a1.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f'{row[key]:.4f}' if isinstance(row[key], float) else row[key] for key in fields})

    lines = [
        r'\begin{tabular}{lrrrrrr}',
        r'\toprule',
        r'Mode & Cả hai đúng & Chỉ ResNet & Chỉ DeiT & Cả hai sai & Jaccard lỗi & Sai cùng lớp \\',
        r'\midrule',
    ]
    for row in rows:
        lines.append(' & '.join((
            row['mode'],
            *(f'{100 * row[key]:.1f}' for key in ('both_right', 'only_resnet', 'only_deit', 'both_wrong')),
            f"{row['error_jaccard']:.2f}",
            f"{100 * row['same_wrong_class']:.1f}",
        )) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    (results_dir / 'agreement_a1.tex').write_text('\n'.join(lines))


def plot_overlap(runs, labels, figures_dir):
    names = list(runs)
    wrong = {name: runs[name]['pred'] != labels for name in names}
    matrix = np.array([[jaccard(wrong[a], wrong[b]) for b in names] for a in names])
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
    axis.set_xticks(range(len(names)), names, rotation=45, ha='right', fontsize=7)
    axis.set_yticks(range(len(names)), names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            axis.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=6,
                      color='black' if matrix[i, j] > 0.6 else 'white')
    axis.set_title('Error overlap on test (Jaccard index of wrong images)', fontsize=9)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(Path(figures_dir) / 'error_overlap_a1.png', dpi=300)
    plt.close(figure)
    return names, matrix


def class_differences(labels, a, b, names):
    '''Per class: ResNet-50 and DeiT-S accuracy, their difference and an exact McNemar p.'''
    rows = []
    for index, name in enumerate(names):
        members = labels == index
        right_a = a['pred'][members] == index
        right_b = b['pred'][members] == index
        _, _, p_value = mcnemar_exact(right_b, right_a)
        rows.append({
            'class': name,
            'acc_resnet': float(right_a.mean()),
            'acc_deit': float(right_b.mean()),
            'diff_pts': float(100 * (right_b.mean() - right_a.mean())),
            'p_mcnemar': p_value,
        })
    return rows


def write_class_diff(by_mode, results_dir):
    fields = ('mode', 'class', 'acc_resnet', 'acc_deit', 'diff_pts', 'p_mcnemar')
    with (results_dir / 'class_diff_a1.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode, rows in by_mode.items():
            for row in sorted(rows, key=lambda item: -item['diff_pts']):
                writer.writerow({'mode': mode, **{key: f'{value:.4f}' if isinstance(value, float) else value
                                                  for key, value in row.items()}})


def plot_class_diff(rows, mode, figures_dir, top=TOP_CLASSES):
    ordered = sorted(rows, key=lambda item: item['diff_pts'])
    shown = ordered[:top] + ordered[-top:]
    figure, axis = plt.subplots(figsize=(6.4, 6.2))
    colors = ['tab:blue' if row['diff_pts'] < 0 else 'tab:orange' for row in shown]
    axis.barh(range(len(shown)), [row['diff_pts'] for row in shown], color=colors)
    labels = [row['class'] for row in shown]
    axis.set_yticks(range(len(shown)), labels, fontsize=7)
    axis.axvline(0, color='0.3', linewidth=0.8)
    axis.set_xlabel('DeiT-S accuracy minus ResNet-50 accuracy (points)')
    axis.set_title(f'{mode}: classes that favor ResNet-50 (blue) or DeiT-S (orange)', fontsize=8)
    figure.tight_layout()
    figure.savefig(Path(figures_dir) / 'class_diff_a1.png', dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Agreement of the CNN and the Transformer on test.')
    parser.add_argument('--results-dir', default=str(RESULTS_DIR))
    parser.add_argument('--figures-dir', default=str(FIGURES_DIR))
    args = parser.parse_args()
    results_dir, figures_dir = Path(args.results_dir), Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs, reference, labels = {}, None, None
    for model in MODELS:
        for mode in MODES:
            run = f'{model}_{mode}'
            path = results_dir / 'predictions' / f'{run}.csv'
            if not path.is_file():
                print(f'skip {run}: no {path}')
                continue
            ids, run_labels, preds, confidence = read_predictions(path)
            if reference is None:
                reference, labels = ids, run_labels
            elif ids != reference:
                raise SystemExit(f'{run} was scored on other test images than the first run')
            runs[run] = {'pred': preds, 'confidence': confidence}

    if not runs:
        raise SystemExit(f'no predictions under {results_dir / "predictions"}: run analyze.py first')
    names = class_names(reference, labels, int(labels.max()) + 1)

    rows, by_mode = [], {}
    for mode in MODES:
        a, b = runs.get(f'resnet50_{mode}'), runs.get(f'deit_s_{mode}')
        if a is None or b is None:
            continue
        rows.append({'mode': mode, **pair_stats(labels, a, b)})
        by_mode[mode] = class_differences(labels, a, b, names)

    write_agreement(rows, results_dir)
    write_class_diff(by_mode, results_dir)
    order, matrix = plot_overlap(runs, labels, figures_dir)
    if PLOT_MODE in by_mode:
        plot_class_diff(by_mode[PLOT_MODE], PLOT_MODE, figures_dir)

    for row in rows:
        print(f"{row['mode']:8} both right {100 * row['both_right']:5.1f}  only R {100 * row['only_resnet']:4.1f}  "
              f"only D {100 * row['only_deit']:4.1f}  both wrong {100 * row['both_wrong']:5.1f}  "
              f"Jaccard {row['error_jaccard']:.2f}  same wrong {100 * row['same_wrong_class']:4.1f}  "
              f"oracle {100 * row['oracle']:5.1f}  ensemble {100 * row['ensemble']:5.1f}")
    for mode, class_rows in by_mode.items():
        diffs = np.array([row['diff_pts'] for row in class_rows])
        significant = sum(row['p_mcnemar'] < 0.05 for row in class_rows)
        print(f'{mode:8} classes DeiT better {np.sum(diffs > 0):3d}, ResNet better {np.sum(diffs < 0):3d}, '
              f'p<0.05 {significant:3d}')
    if 'partial' in by_mode and 'full' in by_mode:
        partial = np.array([row['diff_pts'] for row in by_mode['partial']])
        full = np.array([row['diff_pts'] for row in by_mode['full']])
        print(f'correlation of per-class differences, partial vs full: {np.corrcoef(partial, full)[0, 1]:.2f}')
    print('error overlap (Jaccard):')
    for name, values in zip(order, matrix):
        print(f'  {name:18}', ' '.join(f'{value:.2f}' for value in values))


if __name__ == '__main__':
    main()
