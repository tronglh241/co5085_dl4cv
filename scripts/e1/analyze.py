import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

os.environ.setdefault('MPLBACKEND', 'Agg')

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from common.config.config import CfgNode
from common.metrics.confusion_matrix import ConfusionMatrix
from common.stamp import parse_stamp
from scripts.tensorboard.extract import extract_run

COLUMNS = (
    'Model', 'Variant', 'Params', 'Best epoch',
    'Train acc (%)', 'Val acc (%)', 'Acc (%)', 'Precision (%)', 'Recall (%)', 'F1 (%)',
)
REQUIRED_METRICS = ('accuracy', 'precision', 'recall', 'f1_score')
MAX_STAMP_GAP_S = 60
NUM_CLASSES = 10
THEME_RED = '#c0392b'
MNIST_MEAN = 0.1308
MNIST_STD = 0.3082
FAILURE_LIMIT = 20


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else _repo_root / path


def _config_paths(raw_paths):
    if raw_paths:
        paths = [_resolve(path) for path in raw_paths]
    else:
        paths = sorted((_repo_root / 'configs' / 'e1').glob('test_*.yml'))

    if not paths:
        raise SystemExit('no E1 test configs found')

    return paths


def _run_name(config_path):
    name = Path(config_path).stem

    if name.startswith('test_'):
        return name[len('test_'):]

    if name.startswith('train_'):
        return name[len('train_'):]

    return name


def _model_variant(config_path):
    run = _run_name(config_path)

    if run.endswith('_aug'):
        return run[:-len('_aug')], 'aug'

    return run, 'base'


def _require_checkpoint(cfg):
    if 'loader' not in cfg or not cfg.loader.path:
        raise ValueError('test config must set loader.path')

    path = _resolve(cfg.loader.path)

    if not path.is_file():
        raise FileNotFoundError(f'checkpoint not found: {path}')

    cfg.loader.path = str(path)
    return path


def _log_dir(run, checkpoint):
    root = _repo_root / 'logs' / 'e1' / f'train_{run}'

    if not root.is_dir():
        raise FileNotFoundError(f'log dir not found: {root}')

    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir() and parse_stamp(path.name) is not None),
        key=lambda path: parse_stamp(path.name),
    )

    if not candidates:
        raise FileNotFoundError(f'no timestamped log dirs under {root}')

    target = parse_stamp(checkpoint.parent.name)

    if target is None:
        raise ValueError(f'checkpoint is not inside a timestamp dir: {checkpoint}')

    nearest = min(candidates, key=lambda path: abs((parse_stamp(path.name) - target).total_seconds()))
    gap = abs((parse_stamp(nearest.name) - target).total_seconds())

    if gap > MAX_STAMP_GAP_S:
        raise FileNotFoundError(
            f'no log dir within {MAX_STAMP_GAP_S}s of {checkpoint} under {root} '
            f'(nearest {nearest.name}, {gap:.0f}s)'
        )

    return nearest


def _read_series(path):
    series = {}

    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            series.setdefault(row['tag'], []).append(
                (int(row['step']), float(row['value']))
            )

    for values in series.values():
        values.sort()

    return series


def _load_series(run, checkpoint):
    path = extract_run(_log_dir(run, checkpoint), 'metrics.csv', None)
    return _read_series(path)


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


def _require_evaluation_config(cfg):
    if 'evaluator' not in cfg:
        raise ValueError('test config must define evaluator')

    if 'metrics' not in cfg:
        raise ValueError('test config must define metrics')

    missing = [name for name in REQUIRED_METRICS if name not in cfg.metrics]

    if missing:
        raise ValueError(f'test config is missing metrics: {missing}')


def _evaluate(cfg):
    _require_evaluation_config(cfg)
    cfg.loader.attach(cfg.evaluator, cfg)

    for name, metric in cfg.metrics.items():
        metric.attach(cfg.evaluator, name)

    ConfusionMatrix(NUM_CLASSES).attach(cfg.evaluator, 'confusion_matrix')
    return cfg.evaluator.run()


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


def _display_path(path):
    path = Path(path)
    return str(path.relative_to(_repo_root)) if path.is_relative_to(_repo_root) else str(path)


def _architecture_text(record, model, checkpoint):
    run = record['run']
    variant = record['variant']
    params = record['params']
    module = type(model).__module__
    name = type(model).__name__
    checkpoint_text = _display_path(checkpoint)
    lines = [
        f'run: {run}',
        f'model: {module}.{name}',
        f'variant: {variant}',
        f'params: {params}',
        f'checkpoint: {checkpoint_text}',
        '',
        str(model),
        '',
        'parameters:',
    ]

    for name, parameter in model.named_parameters():
        lines.append(f'{name:<40} {str(tuple(parameter.shape)):<24} {parameter.numel()}')

    return '\n'.join(lines) + '\n'


def _record(config_path):
    run = _run_name(config_path)
    model_name, variant = _model_variant(config_path)
    print(f'analyze {run}')

    cfg = CfgNode.load(config_path)
    checkpoint = _require_checkpoint(cfg)
    series = _load_series(run, checkpoint)
    stats = _epoch_stats(series)
    result = _evaluate(cfg)
    model = cfg.model
    failures = _collect_failures(cfg) if model_name == 'cnn' and variant == 'aug' else None

    record = {
        'run': run,
        'model': model_name,
        'variant': variant,
        'params': sum(parameter.numel() for parameter in model.parameters()),
        'best_epoch': stats['best_epoch'],
        'train_acc': stats['train_acc'],
        'val_acc': stats['val_acc'],
        'accuracy': result.get('accuracy'),
        'precision': result.get('precision'),
        'recall': result.get('recall'),
        'f1': result.get('f1_score'),
        'series': series,
        'failures': failures,
    }

    return record, result['confusion_matrix'], _architecture_text(record, model, checkpoint)


def _number(value):
    return '' if value is None else str(value)


def _percent(value):
    if value is None:
        return ''

    return f'{value * 100:.3f}'


def _csv_row(record):
    return {
        'Model': record['model'],
        'Variant': record['variant'],
        'Params': _number(record['params']),
        'Best epoch': _number(record['best_epoch']),
        'Train acc (%)': _percent(record['train_acc']),
        'Val acc (%)': _percent(record['val_acc']),
        'Acc (%)': _percent(record['accuracy']),
        'Precision (%)': _percent(record['precision']),
        'Recall (%)': _percent(record['recall']),
        'F1 (%)': _percent(record['f1']),
    }


def _write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(_csv_row(record) for record in records)


def _write_confusion_text(path, matrix):
    counts = matrix.cpu().numpy().astype(int)
    labels = range(counts.shape[0])
    lines = ['true\\pred' + ''.join(f'{label:7d}' for label in labels)]

    for label, row in enumerate(counts):
        lines.append(f'{label:<9d}' + ''.join(f'{value:7d}' for value in row))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')


def _write_architecture(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _plot_confusions(runs, figures_dir):
    models = ('softmax', 'mlp', 'cnn')
    labels = {'softmax': 'Softmax', 'mlp': 'MLP', 'cnn': 'CNN'}
    variants = ('base', 'aug')
    variant_mapping = {'base': 'Phiên bản base', 'aug': 'Phiên bản aug'}
    figure, axes = plt.subplots(2, 3, figsize=(11.2, 7.2), sharex=False, sharey=False)
    cmap = LinearSegmentedColormap.from_list('theme_reds', ('#ffffff', '#f5b7b1', THEME_RED))
    cmap.set_bad('white')
    panels = {}

    for column, model in enumerate(models):
        for row, variant in enumerate(variants):
            axis = axes[row, column]
            run = next(
                (item for item in runs if item[0]['model'] == model and item[0]['variant'] == variant),
                None,
            )

            if run is None:
                axis.set_visible(False)
                continue

            counts = run[1].cpu().numpy().astype(float)
            totals = counts.sum(axis=1, keepdims=True)
            rates = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0) * 100
            errors = rates.copy()
            np.fill_diagonal(errors, np.nan)
            panels[row, column] = counts, errors

    finite = [value for _, errors in panels.values() for value in errors[np.isfinite(errors)]]
    shared_max = max(finite) if finite else 1.0

    if shared_max <= 0:
        shared_max = 1.0

    norm = PowerNorm(gamma=0.35, vmin=0, vmax=shared_max)

    for (row, column), (counts, errors) in panels.items():
        axis = axes[row, column]
        axis.imshow(
            errors, cmap=cmap, norm=norm, aspect='equal', interpolation='nearest',
        )

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
                    fontsize=8,
                    color=color,
                )

        ticks = range(NUM_CLASSES)
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.tick_params(axis='x', bottom=True, labelbottom=True, labelsize=8, pad=2, colors='black')
        axis.tick_params(axis='y', left=True, labelleft=True, labelsize=8, pad=2, colors='black')

        if row == 0:
            axis.set_title(labels[models[column]], fontsize=13)

        if column == 0:
            axis.set_ylabel('Nhãn thật', fontsize=10, color='black')

        if row == 1:
            axis.set_xlabel('Dự đoán', fontsize=10, color='black')

    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.92, wspace=0.24, hspace=0.26)

    for row, variant in enumerate(variants):
        position = axes[row, 0].get_position()
        figure.text(
            0.012,
            position.y1 + 0.005,
            variant_mapping[variant],
            ha='left',
            va='bottom',
            fontsize=11,
        )

    figure.savefig(figures_dir / 'confusion_e1.png', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(figure)


def _plot_failures(records, figures_dir):
    record = next(
        (item for item in records if item['model'] == 'cnn' and item['variant'] == 'aug'),
        None,
    )

    if record is None or not record.get('failures'):
        print('no CNN-aug failures to plot')
        return

    failures = record['failures']
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
    figure.savefig(figures_dir / 'failures_cnn_aug.png', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(figure)


def _curve_points(record, split, metric):
    points = record['series'].get(f'{split}/{metric}', [])

    if split == 'train' and metric == 'loss' and not points:
        points = record['series'].get('training/loss', [])

    return points


def _plot_curves(records, figures_dir):
    models = ('softmax', 'mlp', 'cnn')
    labels = {'softmax': 'Softmax', 'mlp': 'MLP', 'cnn': 'CNN'}
    variants = ('base', 'aug')
    variant_mapping = {
        'base': 'Phiên bản base',
        'aug': 'Phiên bản aug',
    }
    split_mapping = {
        'train': 'huấn luyện',
        'valid': 'kiểm định',
    }
    metric_mapping = {
        'loss': 'Loss',
        'accuracy': 'Accuracy',
    }
    train_color = THEME_RED
    valid_color = '#2e7d32'
    best_color = '#6a3d9a'
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 7.8), sharex=True)
    handles = {}
    max_epoch = 0
    loss_max = 0.0

    for record in records:
        for metric in ('loss', 'accuracy'):
            for split in ('train', 'valid'):
                for step, value in _curve_points(record, split, metric):
                    max_epoch = max(max_epoch, step)

                    if metric == 'loss':
                        loss_max = max(loss_max, value)

    for column, model in enumerate(models):
        for row, variant in enumerate(variants):
            axis = axes[row, column]
            record = next(
                (item for item in records if item['model'] == model and item['variant'] == variant),
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
                    fontsize=10,
                    color=best_color,
                    bbox={'boxstyle': 'round,pad=0.15', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8},
                )

            for metric, target, line in (('loss', axis, '--'), ('accuracy', accuracy_axis, '-')):
                for split, color in (('train', train_color), ('valid', valid_color)):
                    points = _curve_points(record, split, metric)

                    if not points:
                        continue

                    x = [step for step, _ in points]
                    y = [value * 100 if metric == 'accuracy' else value for _, value in points]
                    label = f'{metric_mapping[metric]} {split_mapping[split]}'
                    handle, = target.plot(x, y, linestyle=line, color=color, label=label, linewidth=1.1)
                    handles.setdefault(label, handle)

            axis.set_ylim(0, loss_max * 1.05)
            axis.set_xlim(0, max_epoch + 1)
            axis.grid(alpha=0.25)
            accuracy_axis.set_ylim(75, 102)
            accuracy_axis.set_xlim(0, max_epoch + 1)
            accuracy_axis.grid(False)
            axis.tick_params(axis='y', left=column == 0, labelleft=column == 0, labelsize=11, colors='black')
            axis.tick_params(axis='x', labelsize=10, colors='black')
            accuracy_axis.tick_params(
                axis='y', right=column == 2, labelright=column == 2, labelsize=11, colors='black',
            )

            if column == 0:
                axis.set_ylabel('Loss', color='black', fontsize=12)

            if column == 2:
                accuracy_axis.set_ylabel('Accuracy (%)', color='black', fontsize=12)

            if row == 0:
                axis.set_title(labels[model], fontsize=15)

            if row == 1:
                axis.set_xlabel('Epoch', fontsize=11)

    figure.legend(handles.values(), handles.keys(), loc='lower center', ncol=4, fontsize=10, frameon=False)
    figure.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.92, wspace=0.30, hspace=0.28)

    for row, variant in enumerate(variants):
        position = axes[row, 0].get_position()
        figure.text(0.012, position.y1 + 0.005, variant_mapping[variant], ha='left', va='bottom', fontsize=11)

    figure.savefig(figures_dir / 'curves_e1.png', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Write E1 metrics, architectures, curves and confusion matrices.')
    parser.add_argument('--configs', nargs='+', default=None)
    parser.add_argument('--results-dir', default=str(_repo_root / 'reports' / 'results' / 'e1'),
                        help='CSV and text output directory')
    parser.add_argument('--figures-dir', default=str(_repo_root / 'docs' / 'assets' / 'e1'),
                        help='figure output directory')
    args = parser.parse_args()

    config_paths = _config_paths(args.configs)
    runs = [_record(path) for path in config_paths]
    runs.sort(key=lambda item: (item[0]['model'], item[0]['variant'] != 'base'))

    records = [record for record, _, _ in runs]
    results_dir = _resolve(args.results_dir)
    figures_dir = _resolve(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    path = results_dir / 'results_e1.csv'
    _write_csv(path, records)

    for record, matrix, architecture in runs:
        run = record['run']
        _write_architecture(results_dir / f'architecture_{run}.txt', architecture)
        _write_confusion_text(results_dir / f'confusion_{run}.txt', matrix)

    _plot_curves(records, figures_dir)
    _plot_confusions(runs, figures_dir)
    _plot_failures(records, figures_dir)
    print(
        f'wrote {path}, architectures, curves, confusion matrices '
        f'and CNN-aug failures for {len(records)} runs'
    )


if __name__ == '__main__':
    main()
