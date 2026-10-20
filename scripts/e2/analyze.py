'''Build the E2 comparison table, learning curves and confusion matrix.

Each row describes one checkpoint: the TensorBoard run is the one created
together with the checkpoint in loader.path, and the validation accuracy is
read at the epoch BestCheckpoint kept (lowest validation loss).
'''

import argparse
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

from common.config.config import CfgNode
from common.metrics.confusion_matrix import ConfusionMatrix
from models.e2.utils import count_params

COLUMNS = (
    'Model', 'Variant', 'Params', 'Best epoch', 'Time/epoch', 'Val acc', 'Test acc', 'Images/s',
)
STAMP_FORMAT = '%y%m%d%H%M%S'
# TensorBoardLogger and BestCheckpoint stamp their folders a moment apart
# while the same config is built, so allow a small gap when pairing them.
MAX_STAMP_GAP_S = 60
RESULTS_DIR = _repo_root / 'reports' / 'results'


def _run_name(config_path):
    name = Path(config_path).stem
    if name.startswith('test_'):
        name = name[len('test_'):]
    if name.endswith('_torch'):
        return name[:-len('_torch')], 'torch'
    if name.endswith('_scratch'):
        return name[:-len('_scratch')], 'scratch'
    raise ValueError(f'cannot parse run name from {config_path}')


def _stamp(name):
    try:
        return datetime.strptime(name, STAMP_FORMAT)
    except ValueError:
        return None


def _log_dir_for(run, checkpoint_path):
    '''Pick the TensorBoard run that produced checkpoint_path.

    Without a checkpoint, fall back to the newest run of this config.
    '''
    root = _repo_root / 'logs' / 'e2' / f'train_{run}'
    if not root.is_dir():
        return None
    candidates = sorted(path for path in root.iterdir() if path.is_dir() and _stamp(path.name) is not None)
    if not candidates:
        return None
    if checkpoint_path is None:
        return candidates[-1]

    target = _stamp(Path(checkpoint_path).parent.name)
    if target is None:
        print(f'warn {run}: checkpoint folder has no timestamp, using newest log run')
        return candidates[-1]

    nearest = min(candidates, key=lambda path: abs((_stamp(path.name) - target).total_seconds()))
    gap = abs((_stamp(nearest.name) - target).total_seconds())
    if gap > MAX_STAMP_GAP_S:
        print(f'warn {run}: no log run within {MAX_STAMP_GAP_S}s of checkpoint {checkpoint_path}')
        return None
    return nearest


def _extract_metrics(log_dir):
    '''Re-export scalars every time so a retrained run never reads a stale CSV.'''
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


def _mean(values):
    return sum(values) / len(values) if values else None


def _scalar_stats(series):
    '''Validation accuracy at the lowest validation loss, as BestCheckpoint keeps it.

    BestCheckpoint replaces only on a strictly lower loss, so ties go to the
    earliest epoch, which is what min() returns.
    '''
    loss = series.get('valid/loss', [])
    accuracy = dict(series.get('valid/accuracy', []))
    epoch_time = [value for _, value in series.get('training/epoch_time', [])]
    best_epoch = min(loss, key=lambda item: item[1])[0] if loss else None
    return {
        'best_epoch': best_epoch,
        'val_acc': accuracy.get(best_epoch),
        'time_per_epoch': _mean(epoch_time),
        'series': series,
    }


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _throughput(model, loader, warmup=5, repeats=50):
    '''Images/s of the model alone.

    One test batch stays on the device and is fed repeatedly. Iterating the
    DataLoader instead (num_workers=0, per-image CPU transforms) caps every
    model at the same loader speed and hides the model cost.
    '''
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


def _checkpoint_path(cfg):
    path = cfg.loader.path if 'loader' in cfg else None
    if not path or 'REPLACE_WITH_BEST' in str(path):
        return None
    if not Path(path).is_file():
        print(f'skip test, checkpoint missing: {path}')
        return None
    return path


def _evaluate(cfg):
    cfg.loader.attach(cfg.evaluator, cfg)
    if 'metrics' in cfg:
        for name, metric in cfg.metrics.items():
            metric.attach(cfg.evaluator, name)
    matrix = ConfusionMatrix(10)
    matrix.attach(cfg.evaluator, 'confusion_matrix')
    result = cfg.evaluator.run()
    result['images_per_s'] = _throughput(cfg.model, cfg.evaluator.data)
    return result


def _fmt(value, digits=4):
    if value is None:
        return ''
    if isinstance(value, int):
        return str(value)
    return f'{value:.{digits}f}'


def _csv_row(record):
    return {
        'Model': record['family'],
        'Variant': record['variant'],
        'Params': record['params'],
        'Best epoch': _fmt(record['best_epoch']),
        'Time/epoch': _fmt(record['time_per_epoch']),
        'Val acc': _fmt(record['val_acc']),
        'Test acc': _fmt(record['test_acc']),
        'Images/s': _fmt(record['images_per_s'], digits=1),
    }


def _write_latex(records, path):
    '''Write a booktabs tabular that reports/e2.tex pulls in with \\input.'''

    def cell(value, digits=2, scale=1.0):
        if value is None:
            return '--'
        if isinstance(value, int):
            return f'{value:,}'.replace(',', '\\,')
        return f'{value * scale:.{digits}f}'

    lines = [
        r'\begin{tabular}{llrrrrrr}',
        r'\toprule',
        r'Tokenizer & MSA & Params & Epoch & Time/epoch (s) & Val acc (\%) & Test acc (\%) & Ảnh/s \\',
        r'\midrule',
    ]
    previous = None
    for record in records:
        if previous is not None and record['family'] != previous:
            lines.append(r'\addlinespace')
        previous = record['family']
        lines.append(' & '.join((
            record['family'].replace('_', r'\_'),
            record['variant'],
            cell(record['params']),
            cell(record['best_epoch']),
            cell(record['time_per_epoch']),
            cell(record['val_acc'], scale=100.0),
            cell(record['test_acc'], scale=100.0),
            cell(record['images_per_s'], digits=0),
        )) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    path.write_text('\n'.join(lines))


def _plot_curves(groups, out_dir):
    '''Dashed is the train split. The loss panel uses training/loss when that tag exists.'''
    for family, runs in groups.items():
        if not any(item['series'] for item in runs.values()):
            continue
        figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        for variant, color in (('scratch', 'C0'), ('torch', 'C1')):
            item = runs.get(variant)
            if item is None:
                continue
            for axis, tag in ((axes[0], 'loss'), (axes[1], 'accuracy')):
                for split, line in (('train', '--'), ('valid', '-')):
                    key = f'{split}/{tag}'
                    label = f'{split}/{variant}'
                    # training/loss is the optimized loss (dropout on). train/loss
                    # is a later eval pass. Old logs only have the eval pass.
                    if tag == 'loss' and split == 'train' and item['series'].get('training/loss'):
                        key = 'training/loss'
                        label = f'optim/{variant}'
                    points = item['series'].get(key, [])
                    if not points:
                        continue
                    axis.plot(
                        [step for step, _ in points],
                        [value for _, value in points],
                        linestyle=line,
                        color=color,
                        label=label,
                    )
        axes[0].set_title('Loss')
        axes[1].set_title('Accuracy')
        for axis in axes:
            axis.set_xlabel('Epoch')
            axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(out_dir / f'curves_{family}.pdf', dpi=150)
        plt.close(figure)


def _plot_acc_vs_time(records, out_dir):
    points = [item for item in records if item['time_per_epoch'] is not None and item['val_acc'] is not None]
    if not points:
        return
    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    markers = 'osD^vP*X'
    for index, item in enumerate(points):
        axis.scatter(
            item['time_per_epoch'], item['val_acc'],
            marker=markers[index % len(markers)], s=45,
            label=f"{item['family']}/{item['variant']}",
        )
    axis.set_xlabel('Time / epoch (s)')
    axis.set_ylabel('Val accuracy (best-loss epoch)')
    axis.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(out_dir / 'acc_vs_time.pdf', dpi=150)
    plt.close(figure)


def _plot_confusion(result, label, out_dir):
    '''Row-normalised errors only.

    The diagonal (~1000 per class) would swamp the colour scale, so it is
    masked; per-class accuracy goes in the y tick labels and each error cell
    is annotated with its raw count.
    '''
    if result is None or 'confusion_matrix' not in result:
        return
    counts = result['confusion_matrix'].cpu().numpy().astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    rates = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0) * 100
    errors = rates.copy()
    np.fill_diagonal(errors, np.nan)
    classes = range(counts.shape[0])

    figure, axis = plt.subplots(figsize=(5.4, 4.6))
    image = axis.imshow(errors, cmap='Reds', vmin=0)
    figure.colorbar(image, ax=axis, label='% of true class')
    for row in classes:
        for col in classes:
            if row != col and counts[row, col] > 0:
                dark = errors[row, col] > np.nanmax(errors) / 2
                axis.text(col, row, int(counts[row, col]), ha='center', va='center',
                          fontsize=7, color='white' if dark else 'black')
    axis.set_xticks(list(classes))
    axis.set_yticks(list(classes))
    axis.set_yticklabels([f'{row} ({rates[row, row]:.1f}%)' for row in classes], fontsize=8)
    axis.set_xlabel('Predicted')
    axis.set_ylabel('True (per-class accuracy)')
    total_errors = int(counts.sum() - np.trace(counts))
    axis.set_title(f'{label}: {total_errors} errors / {int(counts.sum())}', fontsize=9)
    figure.tight_layout()
    figure.savefig(out_dir / 'confusion_best.pdf', dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Compare E2 runs.')
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--out-dir', default=str(_repo_root / 'reports' / 'images' / 'e2'))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    groups = {}
    for config_path in args.configs:
        family, variant = _run_name(config_path)
        run = f'{family}_{variant}'
        print(f'analyze {run}')
        cfg = CfgNode.load(config_path)
        checkpoint = _checkpoint_path(cfg)
        result = _evaluate(cfg) if checkpoint is not None else None
        log_dir = _log_dir_for(run, checkpoint)
        stats = _scalar_stats(_read_series(_extract_metrics(log_dir)))
        records.append({
            'family': family,
            'variant': variant,
            'run': run,
            'params': count_params(cfg.model),
            'best_epoch': stats['best_epoch'],
            'time_per_epoch': stats['time_per_epoch'],
            'val_acc': stats['val_acc'],
            'test_acc': None if result is None else result.get('accuracy'),
            'images_per_s': None if result is None else result.get('images_per_s'),
            'result': result,
        })
        groups.setdefault(family, {})[variant] = {'series': stats['series']}

    with (RESULTS_DIR / 'e2_results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(_csv_row(record) for record in records)
    _write_latex(records, RESULTS_DIR / 'results_e2.tex')
    _plot_curves(groups, out_dir)
    _plot_acc_vs_time(records, out_dir)

    evaluated = [record for record in records if record['result'] is not None]
    if evaluated:
        best = max(evaluated, key=lambda record: -1.0 if record['val_acc'] is None else record['val_acc'])
        _plot_confusion(best['result'], best['run'], out_dir)
    print(f'wrote {RESULTS_DIR / "e2_results.csv"} and {RESULTS_DIR / "results_e2.tex"}')


if __name__ == '__main__':
    main()
