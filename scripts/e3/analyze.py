import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

from common.config.config import CfgNode
from common.metrics.confusion_matrix import ConfusionMatrix
from common.stamp import parse_stamp

COLUMNS = (
    'Model', 'Sequence', 'Params', 'Best epoch',
    'Train acc (%)', 'Val acc (%)', 'Test acc (%)', 'Images/s',
)
MAX_STAMP_GAP_S = 60
NUM_CLASSES = 10
THEME_RED = '#c0392b'
MNIST_MEAN = 0.1308
MNIST_STD = 0.3082
FAILURE_LIMIT = 20
SEQUENCE_LABELS = {'row': 'hàng', 'column': 'cột'}
CELL_LABELS = {'lstm': 'LSTM', 'gru': 'GRU'}
CELLS = ('lstm', 'gru')
SEQUENCES = ('row', 'column')


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else _repo_root / path


def _config_paths(raw_paths):
    if raw_paths:
        paths = [_resolve(path) for path in raw_paths]
    else:
        paths = sorted((_repo_root / 'configs' / 'e3').glob('test_*.yml'))

    if not paths:
        raise SystemExit('no E3 test configs found')

    return paths


def _run_name(config_path):
    name = Path(config_path).stem
    if name.startswith('test_'):
        name = name[len('test_'):]
    parts = name.split('_')
    if len(parts) != 2 or parts[0] not in CELL_LABELS or parts[1] not in SEQUENCE_LABELS:
        raise ValueError(f'cannot parse E3 run name from {config_path}')
    return parts[0], parts[1]


def _checkpoint_path(cfg):
    path = cfg.loader.path if 'loader' in cfg else None
    if not path or 'REPLACE_WITH_BEST' in str(path):
        return None
    path = _resolve(path)
    if not path.is_file():
        print(f'skip test, checkpoint missing: {path}')
        return None
    cfg.loader.path = str(path)
    return path


def _log_dir_for(cell, sequence, checkpoint_path):
    root = _repo_root / 'logs' / 'e3' / f'train_{cell}_{sequence}'
    if not root.is_dir():
        return None
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir() and parse_stamp(path.name) is not None),
        key=lambda path: parse_stamp(path.name),
    )
    if not candidates:
        return None
    if checkpoint_path is None:
        return candidates[-1]

    target = parse_stamp(Path(checkpoint_path).parent.name)
    if target is None:
        print(f'warn {cell}_{sequence}: checkpoint folder has no timestamp, using newest log run')
        return candidates[-1]

    nearest = min(candidates, key=lambda path: abs((parse_stamp(path.name) - target).total_seconds()))
    gap = abs((parse_stamp(nearest.name) - target).total_seconds())
    if gap > MAX_STAMP_GAP_S:
        print(f'warn {cell}_{sequence}: no log run within {MAX_STAMP_GAP_S}s of checkpoint')
        return None
    return nearest


def _extract_metrics(log_dir):
    if log_dir is None:
        return None
    from scripts.tensorboard.extract import extract_run

    return extract_run(log_dir, 'metrics.csv', None)


def _read_series(csv_path):
    series = {}
    if csv_path is None:
        return series
    with csv_path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            series.setdefault(row['tag'], []).append((int(row['step']), float(row['value'])))
    for values in series.values():
        values.sort(key=lambda item: item[0])
    return series


def _epoch_stats(series):
    loss = series.get('valid/loss', [])
    best_epoch = min(loss, key=lambda item: (item[1], item[0]))[0] if loss else None
    train_accuracy = dict(series.get('train/accuracy', []))
    valid_accuracy = dict(series.get('valid/accuracy', []))
    return {
        'best_epoch': best_epoch,
        'train_acc': train_accuracy.get(best_epoch),
        'val_acc': valid_accuracy.get(best_epoch),
    }


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _throughput(model, loader, warmup=5, repeats=50):
    device = next(model.parameters()).device
    model.eval()
    images = next(iter(loader))[0].to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(images)
        _sync(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(images)
        _sync(device)
    elapsed = time.perf_counter() - start
    return repeats * images.shape[0] / elapsed if elapsed else None


def _evaluate(cfg):
    cfg.loader.attach(cfg.evaluator, cfg)
    for name, metric in cfg.metrics.items():
        metric.attach(cfg.evaluator, name)
    ConfusionMatrix(NUM_CLASSES).attach(cfg.evaluator, 'confusion_matrix')
    result = cfg.evaluator.run()
    result['images_per_s'] = _throughput(cfg.model, cfg.evaluator.data)
    return result


def _collect_failures(cfg, limit=FAILURE_LIMIT):
    model = cfg.evaluator.model
    device = cfg.evaluator.device
    model.eval()
    failures = []
    with torch.no_grad():
        for images, labels in cfg.evaluator.data:
            images = images.to(device)
            labels = labels.to(device)
            probabilities = model(images).softmax(dim=1)
            predicted = probabilities.argmax(dim=1)
            wrong = predicted != labels
            for image, true, pred, confidence in zip(
                images[wrong],
                labels[wrong],
                predicted[wrong],
                probabilities[wrong].max(dim=1).values,
            ):
                failures.append({
                    'image': image.cpu(),
                    'true': int(true),
                    'pred': int(pred),
                    'confidence': float(confidence),
                })
    failures.sort(key=lambda failure: failure['confidence'], reverse=True)
    return failures[:limit]


def _fmt(value, digits=4):
    if value is None:
        return ''
    if isinstance(value, int):
        return str(value)
    return f'{value:.{digits}f}'


def _csv_row(record):
    return {
        'Model': CELL_LABELS[record['cell']],
        'Sequence': record['sequence'],
        'Params': record['params'],
        'Best epoch': _fmt(record['best_epoch']),
        'Train acc (%)': _fmt(record['train_acc'] * 100 if record['train_acc'] is not None else None, 3),
        'Val acc (%)': _fmt(record['val_acc'] * 100 if record['val_acc'] is not None else None, 3),
        'Test acc (%)': _fmt(record['test_acc'] * 100 if record['test_acc'] is not None else None, 3),
        'Images/s': _fmt(record['images_per_s'], 1),
    }


def _curve_points(series, split, metric):
    points = series.get(f'{split}/{metric}', [])
    if split == 'train' and metric == 'loss' and not points:
        points = series.get('training/loss', [])
    return points


def _plot_curves(records, figures_dir):
    train_color = THEME_RED
    valid_color = '#2e7d32'
    best_color = '#6a3d9a'
    loss_max = 0.0
    max_epoch = 0

    for record in records:
        for metric in ('loss', 'accuracy'):
            for split in ('train', 'valid'):
                for step, value in _curve_points(record['series'], split, metric):
                    max_epoch = max(max_epoch, step)
                    if metric == 'loss':
                        loss_max = max(loss_max, value)

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    handles = {}

    for row, sequence in enumerate(SEQUENCES):
        for column, cell in enumerate(CELLS):
            axis = axes[row, column]
            record = next(
                (item for item in records if item['cell'] == cell and item['sequence'] == sequence),
                None,
            )
            if record is None:
                axis.set_visible(False)
                continue

            accuracy_axis = axis.twinx()
            best_epoch = record.get('best_epoch')
            if best_epoch is not None:
                axis.axvline(best_epoch, color=best_color, linestyle=':', linewidth=1.0, zorder=1)
                axis.annotate(
                    f'{best_epoch}',
                    xy=(best_epoch, 0),
                    xycoords=('data', 'axes fraction'),
                    xytext=(0, 4),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color=best_color,
                    bbox={'boxstyle': 'round,pad=0.15', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8},
                )

            for metric, target, line in (('loss', axis, '--'), ('accuracy', accuracy_axis, '-')):
                for split, color in (('train', train_color), ('valid', valid_color)):
                    points = _curve_points(record['series'], split, metric)
                    if not points:
                        continue
                    x = [step for step, _ in points]
                    y = [value * 100 if metric == 'accuracy' else value for _, value in points]
                    split_label = 'huấn luyện' if split == 'train' else 'kiểm định'
                    label = f'{metric.capitalize()} {split_label}'
                    handle, = target.plot(x, y, linestyle=line, color=color, label=label, linewidth=1.1)
                    handles.setdefault(label, handle)

            axis.set_ylim(0, max(loss_max * 1.05, 0.1))
            axis.set_xlim(0, max_epoch + 1)
            axis.grid(alpha=0.25)
            accuracy_axis.set_ylim(75, 102)
            accuracy_axis.set_xlim(0, max_epoch + 1)
            accuracy_axis.grid(False)
            axis.tick_params(axis='y', left=column == 0, labelleft=column == 0, labelsize=10)
            axis.tick_params(axis='x', labelsize=9)
            accuracy_axis.tick_params(
                axis='y', right=column == 1, labelright=column == 1, labelsize=10,
            )

            if column == 0:
                axis.set_ylabel('Loss')
            if column == 1:
                accuracy_axis.set_ylabel('Accuracy (%)')
            if row == 1:
                axis.set_xlabel('Epoch')
            axis.set_title(f'{CELL_LABELS[cell]} · {SEQUENCE_LABELS[sequence]}', fontsize=13)

    figure.legend(handles.values(), handles.keys(), loc='lower center', ncol=4, fontsize=9, frameon=False)
    figure.subplots_adjust(left=0.08, right=0.92, bottom=0.11, top=0.93, wspace=0.30, hspace=0.26)
    figure.savefig(figures_dir / 'curves_e3.png', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(figure)


def _plot_confusions(runs, figures_dir):
    cmap = LinearSegmentedColormap.from_list('theme_reds', ('#ffffff', '#f5b7b1', THEME_RED))
    cmap.set_bad('white')
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 8.2), sharex=False, sharey=False)
    panels = {}

    for row, sequence in enumerate(SEQUENCES):
        for column, cell in enumerate(CELLS):
            axis = axes[row, column]
            run = next(
                (item for item in runs if item[0]['cell'] == cell and item[0]['sequence'] == sequence),
                None,
            )
            if run is None or run[1] is None:
                axis.set_visible(False)
                continue
            counts = run[1].cpu().numpy().astype(float)
            totals = counts.sum(axis=1, keepdims=True)
            rates = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0) * 100
            errors = rates.copy()
            np.fill_diagonal(errors, np.nan)
            panels[row, column] = counts, errors, run[0]

    finite = [value for _, errors, _ in panels.values() for value in errors[np.isfinite(errors)]]
    shared_max = max(finite) if finite else 1.0
    if shared_max <= 0:
        shared_max = 1.0
    norm = PowerNorm(gamma=0.35, vmin=0, vmax=shared_max)

    for (row, column), (counts, errors, record) in panels.items():
        axis = axes[row, column]
        axis.imshow(errors, cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')

        for true_label in range(NUM_CLASSES):
            for predicted_label in range(NUM_CLASSES):
                count = counts[true_label, predicted_label]
                if true_label == predicted_label or count == 0:
                    continue
                color = 'white' if norm(errors[true_label, predicted_label]) > 0.5 else 'black'
                axis.text(
                    predicted_label,
                    true_label,
                    f'{int(count)}',
                    ha='center',
                    va='center',
                    fontsize=7,
                    color=color,
                )

        ticks = range(NUM_CLASSES)
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.tick_params(axis='x', labelsize=7, pad=2)
        axis.tick_params(axis='y', labelsize=7, pad=2)
        errors_total = int(counts.sum() - np.trace(counts))
        axis.set_title(
            f"{CELL_LABELS[record['cell']]} · {SEQUENCE_LABELS[record['sequence']]}"
            f' ({errors_total} lỗi)',
            fontsize=11,
        )
        if column == 0:
            axis.set_ylabel('Nhãn thật')
        if row == 1:
            axis.set_xlabel('Dự đoán')

    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.07, top=0.94, wspace=0.24, hspace=0.26)
    figure.savefig(figures_dir / 'confusion_e3.png', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(figure)


def _plot_failures(record, figures_dir):
    failures = record.get('failures')
    if not failures:
        print('no failures to plot')
        return

    columns = 5
    rows = (len(failures) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(9.0, 1.75 * rows + 0.35), squeeze=False)

    for axis, failure in zip(axes.flat, failures):
        image = (failure['image'] * MNIST_STD + MNIST_MEAN).clamp(0, 1)
        axis.imshow(image[0].numpy(), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axis.set_title(
            f"{failure['true']}→{failure['pred']} ({failure['confidence']:.2f})",
            fontsize=9,
            pad=3,
        )
        axis.set_axis_off()

    for axis in axes.flat[len(failures):]:
        axis.set_axis_off()

    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.97, wspace=0.08, hspace=0.30)
    figure.savefig(figures_dir / 'failures_best.png', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(figure)


def _record(config_path):
    cell, sequence = _run_name(config_path)
    run = f'{cell}_{sequence}'
    print(f'analyze {run}')

    cfg = CfgNode.load(config_path)
    checkpoint = _checkpoint_path(cfg)
    result = _evaluate(cfg) if checkpoint is not None else None
    series = _read_series(_extract_metrics(_log_dir_for(cell, sequence, checkpoint)))
    stats = _epoch_stats(series)
    params = sum(parameter.numel() for parameter in cfg.model.parameters())

    record = {
        'cell': cell,
        'sequence': sequence,
        'run': run,
        'params': params,
        'best_epoch': stats['best_epoch'],
        'train_acc': stats['train_acc'],
        'val_acc': stats['val_acc'],
        'test_acc': None if result is None else result.get('accuracy'),
        'images_per_s': None if result is None else result.get('images_per_s'),
        'series': series,
        'failures': None,
    }
    matrix = None if result is None else result.get('confusion_matrix')
    return record, matrix, cfg, checkpoint


def main():
    parser = argparse.ArgumentParser(description='Compare E3 recurrent runs.')
    parser.add_argument('--configs', nargs='+', default=None)
    parser.add_argument('--results-dir', default=str(_repo_root / 'reports' / 'results' / 'e3'),
                        help='CSV output directory')
    parser.add_argument('--figures-dir', default=str(_repo_root / 'docs' / 'assets' / 'e3'),
                        help='figure output directory')
    args = parser.parse_args()

    runs = [_record(path) for path in _config_paths(args.configs)]
    runs.sort(key=lambda item: (SEQUENCES.index(item[0]['sequence']), CELLS.index(item[0]['cell'])))
    records = [record for record, _, _, _ in runs]

    results_dir = _resolve(args.results_dir)
    figures_dir = _resolve(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    path = results_dir / 'results_e3.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(_csv_row(record) for record in records)

    _plot_curves(records, figures_dir)
    _plot_confusions(runs, figures_dir)

    evaluated = [item for item in runs if item[0]['test_acc'] is not None]
    if evaluated:
        best = max(evaluated, key=lambda item: item[0]['val_acc'] if item[0]['val_acc'] is not None else -1.0)
        best[0]['failures'] = _collect_failures(best[2])
        _plot_failures(best[0], figures_dir)

    print(f'wrote {path} and figures to {figures_dir}')


if __name__ == '__main__':
    main()
