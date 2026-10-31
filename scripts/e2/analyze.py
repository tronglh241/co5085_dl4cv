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
from models.e2.utils import count_params

COLUMNS = (
    'Model', 'Variant', 'Params', 'Best epoch', 'Time/epoch', 'Train acc', 'Val acc', 'Test acc', 'Errors', 'Images/s',
)
# TensorBoardLogger and BestCheckpoint share one stamp per run now. Runs made
# before that stamped their folders a moment apart, so allow a small gap.
MAX_STAMP_GAP_S = 60
NUM_CLASSES = 10
MNIST_MEAN = 0.1308
MNIST_STD = 0.3082
FAILURE_LIMIT = 20
# Same palette as scripts/e1/analyze.py, so E1 and E2 figures read alike.
THEME_RED = '#c0392b'
VALID_GREEN = '#2e7d32'
BEST_PURPLE = '#6a3d9a'
CURVE_FAMILIES = ('patch', 'patch7', 'patch_overlap', 'row', 'cnnstem')
FAMILY_LABELS = {'patch': 'Patch $4\\times4$', 'patch7': 'Patch $7\\times7$',
                 'patch_overlap': 'Patch $8\\times8$ chồng lấn', 'row': 'Hàng', 'cnnstem': 'CNN-stem'}
VARIANT_LABELS = {'scratch': 'MSA tự cài đặt', 'torch': 'MSA PyTorch'}


def _run_name(config_path):
    name = Path(config_path).stem
    if name.startswith('test_'):
        name = name[len('test_'):]
    if name.endswith('_torch'):
        return name[:-len('_torch')], 'torch'
    if name.endswith('_scratch'):
        return name[:-len('_scratch')], 'scratch'
    raise ValueError(f'cannot parse run name from {config_path}')


def _log_dir_for(run, checkpoint_path):
    '''Pick the TensorBoard run that produced checkpoint_path.

    Without a checkpoint, fall back to the newest run of this config.
    '''
    root = _repo_root / 'logs' / 'e2' / f'train_{run}'
    if not root.is_dir():
        return None
    candidates = sorted(path for path in root.iterdir() if path.is_dir() and parse_stamp(path.name) is not None)
    if not candidates:
        return None
    if checkpoint_path is None:
        return candidates[-1]

    target = parse_stamp(Path(checkpoint_path).parent.name)
    if target is None:
        print(f'warn {run}: checkpoint folder has no timestamp, using newest log run')
        return candidates[-1]

    nearest = min(candidates, key=lambda path: abs((parse_stamp(path.name) - target).total_seconds()))
    gap = abs((parse_stamp(nearest.name) - target).total_seconds())
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
    train_accuracy = dict(series.get('train/accuracy', []))
    return {
        'best_epoch': best_epoch,
        'train_acc': train_accuracy.get(best_epoch),
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
        'Train acc': _fmt(record['train_acc']),
        'Val acc': _fmt(record['val_acc']),
        'Test acc': _fmt(record['test_acc']),
        'Errors': _fmt(record['errors']),
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


def _curve_points(series, split, metric):
    '''training/loss is the optimized loss (dropout on); train/loss is a later eval
    pass. Old logs only have the eval pass.'''
    if metric == 'loss' and split == 'train' and series.get('training/loss'):
        return series['training/loss']
    return series.get(f'{split}/{metric}', [])


def _plot_curves(groups, figures_dir):
    '''One figure like E1: row = tokenizer, column = MSA variant.

    Dashed is loss (left axis), solid is accuracy (right axis); the dotted line
    marks the epoch whose checkpoint was kept.
    '''
    families = [family for family in CURVE_FAMILIES if family in groups]
    variants = ('scratch', 'torch')
    if not families:
        return
    split_labels = {'train': 'train', 'valid': 'val'}
    colors = {'train': THEME_RED, 'valid': VALID_GREEN}

    max_epoch, loss_max, acc_min = 0, 0.0, 100.0
    for family in families:
        for item in groups[family].values():
            for split in colors:
                for metric in ('loss', 'accuracy'):
                    for step, value in _curve_points(item['series'], split, metric):
                        max_epoch = max(max_epoch, step)
                        if metric == 'loss':
                            loss_max = max(loss_max, value)
                        else:
                            acc_min = min(acc_min, value * 100)
    if not max_epoch:
        return
    acc_floor = max(0.0, np.floor(acc_min / 5) * 5)

    figure, axes = plt.subplots(len(families), 2, figsize=(9.0, 2.5 * len(families) + 0.8),
                                sharex=True, squeeze=False)
    handles = {}
    for row, family in enumerate(families):
        for column, variant in enumerate(variants):
            axis = axes[row, column]
            item = groups[family].get(variant)
            if item is None:
                axis.set_visible(False)
                continue
            accuracy_axis = axis.twinx()
            best_epoch = item.get('best_epoch')
            if best_epoch is not None:
                axis.axvline(best_epoch, color=BEST_PURPLE, linestyle=':', linewidth=1.0, zorder=1)
                axis.annotate(
                    f'{best_epoch}', xy=(best_epoch, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom',
                    fontsize=10, color=BEST_PURPLE,
                    bbox={'boxstyle': 'round,pad=0.15', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8},
                )
            for metric, target, line in (('loss', axis, '--'), ('accuracy', accuracy_axis, '-')):
                for split, color in colors.items():
                    points = _curve_points(item['series'], split, metric)
                    if not points:
                        continue
                    scale = 100 if metric == 'accuracy' else 1
                    label = f"{'Loss' if metric == 'loss' else 'Accuracy'} {split_labels[split]}"
                    handle, = target.plot([step for step, _ in points], [value * scale for _, value in points],
                                          linestyle=line, color=color, label=label, linewidth=1.1)
                    handles.setdefault(label, handle)

            axis.set_ylim(0, loss_max * 1.05)
            axis.set_xlim(0, max_epoch + 1)
            axis.grid(alpha=0.25)
            accuracy_axis.set_ylim(acc_floor, 102)
            accuracy_axis.set_xlim(0, max_epoch + 1)
            accuracy_axis.grid(False)
            axis.tick_params(axis='y', left=column == 0, labelleft=column == 0, labelsize=11, colors='black')
            axis.tick_params(axis='x', labelsize=10, colors='black')
            right_edge = column == 1 or 'torch' not in groups[family]
            accuracy_axis.tick_params(axis='y', right=right_edge, labelright=right_edge,
                                      labelsize=11, colors='black')
            if column == 0:
                axis.set_ylabel('Loss', color='black', fontsize=12)
            if right_edge:
                accuracy_axis.set_ylabel('Accuracy (%)', color='black', fontsize=12)
            if row == 0:
                axis.set_title(VARIANT_LABELS[variant], fontsize=14)
            if row == len(families) - 1:
                axis.set_xlabel('Epoch', fontsize=11)

    figure.legend(handles.values(), handles.keys(), loc='lower center', ncol=4, fontsize=10, frameon=False)
    figure.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.93, wspace=0.12, hspace=0.18)
    for row, family in enumerate(families):
        position = axes[row, 0].get_position()
        figure.text(0.012, position.y1 + 0.005, FAMILY_LABELS[family], ha='left', va='bottom', fontsize=11)
    figure.savefig(figures_dir / 'curves_e2.png', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(figure)


def _plot_acc_vs_time(records, figures_dir):
    points = [item for item in records if item['time_per_epoch'] is not None and item['val_acc'] is not None]
    if not points:
        return
    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    markers = 'osD^vP*X'
    palette = (THEME_RED, VALID_GREEN, BEST_PURPLE, '#e67e22', '#2980b9', '#7f8c8d', '#c2185b', '#00897b')
    for index, item in enumerate(points):
        axis.scatter(
            item['time_per_epoch'], item['val_acc'] * 100,
            marker=markers[index % len(markers)], s=45, color=palette[index % len(palette)],
            label=f"{item['family']}/{item['variant']}",
        )
    axis.set_xlabel('Thời gian mỗi epoch (s)', fontsize=10)
    axis.set_ylabel('Accuracy kiểm định (%)', fontsize=10)
    axis.grid(alpha=0.25)
    axis.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(figures_dir / 'acc_vs_time.png', dpi=300)
    plt.close(figure)


def _plot_confusion(result, label, figures_dir):
    '''Row-normalised errors only, drawn like scripts/e1/analyze.py.

    The diagonal (~1000 per class) would swamp the colour scale, so it is
    masked; each error cell carries its raw count.
    '''
    if result is None or 'confusion_matrix' not in result:
        return
    counts = result['confusion_matrix'].cpu().numpy().astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    rates = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0) * 100
    errors = rates.copy()
    np.fill_diagonal(errors, np.nan)

    cmap = LinearSegmentedColormap.from_list('theme_reds', ('#ffffff', '#f5b7b1', THEME_RED))
    cmap.set_bad('white')
    finite = errors[np.isfinite(errors)]
    vmax = finite.max() if finite.size and finite.max() > 0 else 1.0
    norm = PowerNorm(gamma=0.35, vmin=0, vmax=vmax)

    figure, axis = plt.subplots(figsize=(4.6, 4.4))
    axis.imshow(errors, cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')
    for true_label in range(NUM_CLASSES):
        for predicted_label in range(NUM_CLASSES):
            count = counts[true_label, predicted_label]
            if true_label == predicted_label or count == 0:
                continue
            color = 'white' if norm(errors[true_label, predicted_label]) > 0.5 else 'black'
            axis.text(predicted_label, true_label, f'{int(count)}', ha='center', va='center',
                      fontsize=8, color=color)
    ticks = range(NUM_CLASSES)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.tick_params(axis='both', labelsize=8, pad=2, colors='black')
    axis.set_xlabel('Dự đoán', fontsize=10, color='black')
    axis.set_ylabel('Nhãn thật', fontsize=10, color='black')
    total_errors = int(counts.sum() - np.trace(counts))
    axis.set_title(f'{label}: {total_errors} lỗi / {int(counts.sum())}', fontsize=11)
    figure.tight_layout()
    figure.savefig(figures_dir / 'confusion_best.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(figure)


def _confusion_panel(axis, result, cmap, norm, title=None):
    counts = result['confusion_matrix'].cpu().numpy().astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    rates = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0) * 100
    errors = rates.copy()
    np.fill_diagonal(errors, np.nan)
    axis.imshow(errors, cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')
    for true_label in range(NUM_CLASSES):
        for predicted_label in range(NUM_CLASSES):
            count = counts[true_label, predicted_label]
            if true_label == predicted_label or count == 0:
                continue
            color = 'white' if norm(errors[true_label, predicted_label]) > 0.5 else 'black'
            axis.text(predicted_label, true_label, f'{int(count)}', ha='center', va='center',
                      fontsize=7, color=color)
    axis.set_xticks(range(NUM_CLASSES))
    axis.set_yticks(range(NUM_CLASSES))
    axis.tick_params(axis='both', labelsize=7, pad=2, colors='black')
    if title:
        axis.set_title(title, fontsize=12)
    return errors


def _plot_confusions(records, figures_dir):
    '''All evaluated runs like E1: row = MSA variant, column = tokenizer.'''
    families = [family for family in CURVE_FAMILIES if any(r['family'] == family for r in records)]
    variants = ('scratch', 'torch')
    by_key = {(r['family'], r['variant']): r for r in records if r['result'] is not None}
    if not by_key:
        return
    cmap = LinearSegmentedColormap.from_list('theme_reds', ('#ffffff', '#f5b7b1', THEME_RED))
    cmap.set_bad('white')
    vmax = 1.0
    for record in by_key.values():
        counts = record['result']['confusion_matrix'].cpu().numpy().astype(float)
        rates = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) * 100
        np.fill_diagonal(rates, 0)
        vmax = max(vmax, rates.max())
    norm = PowerNorm(gamma=0.35, vmin=0, vmax=vmax)
    figure, axes = plt.subplots(2, len(families), figsize=(2.6 * len(families) + 0.4, 6.4), squeeze=False)
    for row, variant in enumerate(variants):
        for column, family in enumerate(families):
            axis = axes[row, column]
            record = by_key.get((family, variant))
            if record is None:
                axis.set_visible(False)
                continue
            _confusion_panel(axis, record['result'], cmap, norm, FAMILY_LABELS[family] if row == 0 else None)
            if column == 0:
                axis.set_ylabel('Nhãn thật', fontsize=9, color='black')
            if row == 1:
                axis.set_xlabel('Dự đoán', fontsize=9, color='black')
    figure.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.9, wspace=0.28, hspace=0.22)
    for row, variant in enumerate(variants):
        position = axes[row, 0].get_position()
        figure.text(0.006, position.y1 + 0.005, VARIANT_LABELS[variant], ha='left', va='bottom', fontsize=11)
    figure.savefig(figures_dir / 'confusion_e2.png', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(figure)


def _collect_failures(cfg, limit=FAILURE_LIMIT):
    model = cfg.evaluator.model
    device = cfg.evaluator.device
    model.eval()
    failures = []
    with torch.no_grad():
        for images, labels in cfg.evaluator.data:
            images, labels = images.to(device), labels.to(device)
            probabilities = model(images).softmax(dim=1)
            predicted = probabilities.argmax(dim=1)
            wrong = predicted != labels
            for image, true, pred, confidence in zip(
                images[wrong], labels[wrong], predicted[wrong], probabilities[wrong].max(dim=1).values,
            ):
                failures.append({'image': image.cpu(), 'true': int(true), 'pred': int(pred),
                                 'confidence': float(confidence)})
    failures.sort(key=lambda failure: failure['confidence'], reverse=True)
    return failures[:limit]


def _plot_failures(cfg, figures_dir):
    '''The most confident mistakes of the best model, as in E1.'''
    failures = _collect_failures(cfg)
    if not failures:
        print('no failures to plot')
        return
    columns = 5
    rows = (len(failures) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(9.0, 1.75 * rows + 0.35), squeeze=False)
    for axis, failure in zip(axes.flat, failures):
        image = (failure['image'] * MNIST_STD + MNIST_MEAN).clamp(0, 1)
        axis.imshow(image[0].numpy(), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axis.set_title(f"{failure['true']}→{failure['pred']} ({failure['confidence']:.2f})", fontsize=9, pad=3)
        axis.set_axis_off()
    for axis in axes.flat[len(failures):]:
        axis.set_axis_off()
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.97, wspace=0.08, hspace=0.30)
    figure.savefig(figures_dir / 'failures_best.png', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Compare E2 runs.')
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--figures-dir', default=str(_repo_root / 'docs' / 'assets' / 'e2'),
                        help='figure directory')
    parser.add_argument('--results-dir', default=str(_repo_root / 'reports' / 'results' / 'e2'),
                        help='CSV and LaTeX table directory')
    args = parser.parse_args()
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

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
            'train_acc': stats['train_acc'],
            'val_acc': stats['val_acc'],
            'errors': None if result is None else int(round(float(result['confusion_matrix'].sum() - result['confusion_matrix'].trace()))),
            'cfg': cfg,
            'test_acc': None if result is None else result.get('accuracy'),
            'images_per_s': None if result is None else result.get('images_per_s'),
            'result': result,
        })
        groups.setdefault(family, {})[variant] = {'series': stats['series'], 'best_epoch': stats['best_epoch']}

    with (results_dir / 'results_e2.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(_csv_row(record) for record in records)
    _write_latex(records, results_dir / 'results_e2.tex')
    _plot_curves(groups, figures_dir)
    _plot_acc_vs_time(records, figures_dir)

    evaluated = [record for record in records if record['result'] is not None]
    if evaluated:
        best = max(evaluated, key=lambda record: -1.0 if record['val_acc'] is None else record['val_acc'])
        _plot_confusion(best['result'], best['run'], figures_dir)
        _plot_failures(best['cfg'], figures_dir)
    _plot_confusions(records, figures_dir)
    print(f'wrote {results_dir / "results_e2.csv"} and {results_dir / "results_e2.tex"}')
    print(f'wrote figures to {figures_dir}')


if __name__ == '__main__':
    main()
