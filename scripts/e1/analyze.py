import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator

from common.config.config import CfgNode
from common.metrics.confusion_matrix import ConfusionMatrix

RESULTS_DIR = _repo_root / 'reports' / 'results' / 'e1'
IMAGES = _repo_root / 'docs' / 'assets' / 'e1'
STAMP_FORMAT = '%y%m%d%H%M%S'
MAX_STAMP_GAP_S = 60
MEAN = 0.1308
STD = 0.3082

COLUMNS = (
    'Model', 'Variant', 'Params', 'Best epoch', 'Wall s/epoch',
    'Val acc (%)', 'Test acc (%)', 'Precision (%)', 'Recall (%)', 'F1 (%)', 'Images/s',
)


def run_name(config_path):
    name = Path(config_path).stem

    if name.startswith('test_'):
        return name[len('test_'):]

    if name.startswith('train_'):
        return name[len('train_'):]

    return name


def _stamp(name):
    try:
        return datetime.strptime(name, STAMP_FORMAT)
    except ValueError:
        return None


def require_checkpoint(cfg):
    if 'loader' not in cfg or not cfg.loader.path:
        raise ValueError('test config must set loader.path')

    path = Path(cfg.loader.path)

    if not path.is_absolute():
        path = _repo_root / path

    if not path.is_file():
        raise FileNotFoundError(f'checkpoint not found: {path}')

    cfg.loader.path = str(path)
    return path


def pair_log_dir(run, checkpoint):
    root = _repo_root / 'logs' / 'e1' / f'train_{run}'
    if not root.is_dir():
        raise FileNotFoundError(f'log dir not found: {root}')

    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir() and _stamp(path.name) is not None),
        key=lambda path: _stamp(path.name),
    )
    if not candidates:
        raise FileNotFoundError(f'no timestamped log dirs under {root}')

    target = _stamp(Path(checkpoint).parent.name)
    if target is None:
        raise ValueError(f'checkpoint is not inside a timestamp dir: {checkpoint}')

    nearest = min(candidates, key=lambda path: abs((_stamp(path.name) - target).total_seconds()))
    gap = abs((_stamp(nearest.name) - target).total_seconds())
    if gap > MAX_STAMP_GAP_S:
        raise FileNotFoundError(
            f'no log dir within {MAX_STAMP_GAP_S}s of {checkpoint} under {root} '
            f'(nearest {nearest.name}, {gap:.0f}s)'
        )
    return nearest


def load_series(log_dir):
    accumulator = event_accumulator.EventAccumulator(str(log_dir), size_guidance={'scalars': 0})
    accumulator.Reload()
    series = {}

    for tag in accumulator.Tags()['scalars']:
        latest = {}
        for event in accumulator.Scalars(tag):
            if event.step not in latest or event.wall_time > latest[event.step][0]:
                latest[event.step] = (event.wall_time, event.value)
        series[tag] = [
            (step, wall, value) for step, (wall, value) in sorted(latest.items())
        ]

    return series


def _wall_epoch_time(loss):
    if len(loss) < 2:
        return None
    deltas = [later[1] - earlier[1] for earlier, later in zip(loss, loss[1:])]
    return sum(deltas) / len(deltas)


def epoch_stats(series):
    loss = series.get('valid/loss', [])
    accuracy = {step: value for step, _, value in series.get('valid/accuracy', [])}
    best_epoch = min(loss, key=lambda item: item[2])[0] if loss else None
    epoch_time = [value for _, _, value in series.get('training/epoch_time', [])]
    return {
        'best_epoch': best_epoch,
        'val_acc': accuracy.get(best_epoch),
        'time_per_epoch': (
            sum(epoch_time) / len(epoch_time) if epoch_time else _wall_epoch_time(loss)
        ),
    }


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def throughput(model, loader, warmup=5, repeats=50):
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


def evaluate(cfg, confusion=False):
    cfg.loader.attach(cfg.evaluator, cfg)

    if 'metrics' in cfg:
        for name, metric in cfg.metrics.items():
            metric.attach(cfg.evaluator, name)

    if confusion:
        ConfusionMatrix(10).attach(cfg.evaluator, 'confusion_matrix')

    result = cfg.evaluator.run()
    result['images_per_s'] = throughput(cfg.model, cfg.evaluator.data)
    return result


def collect_errors(cfg, limit=10):
    model = cfg.model
    device = next(model.parameters()).device
    model.eval()
    errors = []

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
                errors.append({
                    'image': image.cpu(),
                    'true': int(true),
                    'pred': int(pred),
                    'confidence': float(confidence),
                })
                if len(errors) == limit:
                    return errors

    return errors


def count_params(model):
    return sum(parameter.numel() for parameter in model.parameters())


def denormalize(images):
    return (images * STD + MEAN).clamp(0, 1)


def draw_confusion(axis, counts, label):
    counts = counts.astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    rates = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0) * 100
    errors = rates.copy()
    np.fill_diagonal(errors, np.nan)
    image = axis.imshow(errors, cmap='Reds', vmin=0)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    finite = errors[np.isfinite(errors)]
    limit = finite.max() if finite.size else 0
    classes = range(counts.shape[0])
    for row in classes:
        for col in classes:
            if row != col and counts[row, col] > 0:
                dark = errors[row, col] > limit / 2
                axis.text(
                    col,
                    row,
                    int(counts[row, col]),
                    ha='center',
                    va='center',
                    fontsize=7,
                    color='white' if dark else 'black',
                )

    axis.set_xticks(list(classes))
    axis.set_yticks(list(classes))
    axis.set_yticklabels([f'{row} ({rates[row, row]:.1f}%)' for row in classes], fontsize=8)
    axis.set_xlabel('Predicted')
    axis.set_ylabel('True (per-class accuracy)')
    total_errors = int(counts.sum() - np.trace(counts))
    axis.set_title(f'{label}: {total_errors} errors / {int(counts.sum())}', fontsize=9)


def _cell(value):
    if value is None:
        return '--'
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: '' if row.get(column) is None else _cell(row.get(column)) for column in columns})


def print_table(columns, rows):
    table = [[_cell(row.get(column)) for column in columns] for row in rows]
    widths = [len(column) for column in columns]
    for line in table:
        for index, value in enumerate(line):
            widths[index] = max(widths[index], len(value))

    print('  '.join(column.ljust(widths[index]) for index, column in enumerate(columns)))
    print('  '.join('-' * width for width in widths))
    for line in table:
        print('  '.join(value.ljust(widths[index]) for index, value in enumerate(line)))


def save_png(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(figure)


def _percent(value):
    return None if value is None else f'{value * 100:.2f}'


def _records():
    records = []
    for config_path in sorted((_repo_root / 'configs' / 'e1').glob('test_*.yml')):
        run = run_name(config_path)
        variant = 'aug' if run.endswith('_aug') else 'base'
        model = run[:-len('_aug')] if variant == 'aug' else run

        print(f'analyze {run}')
        cfg = CfgNode.load(config_path)
        checkpoint = require_checkpoint(cfg)

        series = load_series(pair_log_dir(run, checkpoint))
        stats = epoch_stats(series)
        result = evaluate(cfg, confusion=True)

        records.append({
            'model': model,
            'variant': variant,
            'run': run,
            'params': count_params(cfg.model),
            'best_epoch': stats['best_epoch'],
            'time_per_epoch': stats['time_per_epoch'],
            'val_acc': stats['val_acc'],
            'test_acc': result.get('accuracy'),
            'precision': result.get('precision'),
            'recall': result.get('recall'),
            'f1': result.get('f1_score'),
            'images_per_s': result.get('images_per_s'),
            'result': result,
            'series': series,
            'cfg': cfg,
        })

    return sorted(records, key=lambda record: (record['model'], record['variant'] != 'base'))


def _csv_rows(records):
    return [
        {
            'Model': record['model'],
            'Variant': record['variant'],
            'Params': record['params'],
            'Best epoch': record['best_epoch'],
            'Wall s/epoch': record['time_per_epoch'],
            'Val acc (%)': _percent(record['val_acc']),
            'Test acc (%)': _percent(record['test_acc']),
            'Precision (%)': _percent(record['precision']),
            'Recall (%)': _percent(record['recall']),
            'F1 (%)': _percent(record['f1']),
            'Images/s': record['images_per_s'],
        }
        for record in records
    ]


def _points(series, key):
    return [(step, value) for step, _, value in series.get(key, [])]


def _plot_curves(records):
    models = sorted({record['model'] for record in records})
    figure, axes = plt.subplots(len(models), 2, figsize=(9, 2.7 * len(models)), squeeze=False)

    for row, model in enumerate(models):
        for record in (item for item in records if item['model'] == model):
            color = 'C0' if record['variant'] == 'base' else 'C1'
            train_loss = 'train/loss' if record['series'].get('train/loss') else 'training/loss'
            for axis, train_key, valid_key in (
                (axes[row][0], train_loss, 'valid/loss'),
                (axes[row][1], 'train/accuracy', 'valid/accuracy'),
            ):
                for key, style, split in ((train_key, '--', 'train'), (valid_key, '-', 'valid')):
                    points = _points(record['series'], key)
                    if points:
                        axis.plot(
                            [step for step, _ in points],
                            [value for _, value in points],
                            linestyle=style,
                            color=color,
                            label=f"{record['variant']} {split}",
                        )

        axes[row][0].set_title(f'{model}: loss')
        axes[row][1].set_title(f'{model}: accuracy')
        for axis in axes[row]:
            axis.set_xlabel('Epoch')
            axis.legend(fontsize=7, ncol=2)

    figure.tight_layout()
    save_png(figure, IMAGES / 'curves.png')


def _plot_confusion(records):
    selected = [
        record for record in records
        if record['variant'] == 'base' and 'confusion_matrix' in record['result']
    ]
    if not selected:
        return

    figure, axes = plt.subplots(1, len(selected), figsize=(5.4 * len(selected), 4.6), squeeze=False)
    for axis, record in zip(axes[0], selected):
        draw_confusion(axis, record['result']['confusion_matrix'].cpu().numpy(), record['run'])
    figure.tight_layout()
    save_png(figure, IMAGES / 'confusion.png')


def _plot_errors(records):
    best = max(records, key=lambda record: -1.0 if record['val_acc'] is None else record['val_acc'])
    errors = collect_errors(best['cfg'], limit=10)
    if not errors:
        return

    figure, axes = plt.subplots(2, 5, figsize=(10, 4.6))
    for axis, error in zip(axes.flat, errors):
        axis.imshow(denormalize(error['image'])[0], cmap='gray')
        axis.set_title(
            f"{error['true']} -> {error['pred']} ({error['confidence']:.2f})",
            fontsize=8,
        )
        axis.set_axis_off()
    for axis in axes.flat[len(errors):]:
        axis.set_axis_off()
    figure.tight_layout()
    save_png(figure, IMAGES / 'errors.png')


def main():
    records = _records()
    rows = _csv_rows(records)
    path = RESULTS_DIR / 'e1_results.csv'
    write_csv(path, COLUMNS, rows)
    print_table(COLUMNS, rows)
    _plot_curves(records)
    _plot_confusion(records)
    _plot_errors(records)
    print(f'\nwrote {path} and {IMAGES}')


if __name__ == '__main__':
    main()
