'''Build the A1 tables and figures from the finished runs.

For every run in configs/a1/runs.json that has a best_*.pt:
- evaluate test_<run>.yml on the test split once (accuracy, macro P/R/F1,
  confusion matrix) and keep the per-image predictions,
- read the TensorBoard run with the same stamp as the checkpoint (val curves,
  time/epoch, peak memory),
- count total and trainable parameters of its freezing mode,
- add a Wilson interval for its test accuracy and its per-class accuracy,

then compare the runs on the planned contrasts of scripts/a1/stats.py
(paired bootstrap interval, exact McNemar test, Holm correction).

    python scripts/a1/analyze.py                 # every run that has a checkpoint
    python scripts/a1/analyze.py --runs resnet50_full,deit_s_full   # only some runs
'''

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image

from common.config.config import CfgNode
from common.engine.engine import Events
from common.metrics.confusion_matrix import ConfusionMatrix
from models.a1.classifier import freeze_parameters
from scripts.a1.stats import compare, wilson_interval, write_comparisons
from scripts.tensorboard.extract import extract_run

RUNS_JSON = _repo_root / 'configs' / 'a1' / 'runs.json'
MODES = ('scratch', 'head', 'partial', 'full')
COLUMNS = (
    'run', 'model', 'mode', 'seed', 'params', 'trainable', 'best_epoch', 'val_acc',
    'test_acc', 'test_ci_low', 'test_ci_high', 'test_f1', 'time_per_epoch_s', 'train_time_min',
    'peak_mem_gb', 'images_per_s',
)


def load_runs(selection):
    spec = json.loads(RUNS_JSON.read_text())
    if selection is None:
        names = list(spec['runs'])
    elif selection in spec['sets']:
        names = spec['sets'][selection]
    else:
        names = [name.strip() for name in selection.split(',') if name.strip()]
        unknown = sorted(set(names) - set(spec['runs']))
        if unknown:
            raise SystemExit(f'unknown runs: {unknown}')
    return {name: spec['runs'][name] for name in names}


def latest_best(run):
    '''best_*.pt of the newest stamp folder of this run.'''
    found = sorted((_repo_root / 'checkpoints' / 'a1' / f'train_{run}').glob('*/best_*.pt'),
                   key=lambda path: path.parent.name)
    return found[-1] if found else None


def read_series(run, checkpoint):
    '''Scalars of the TensorBoard run that shares the checkpoint's stamp.'''
    log_dir = _repo_root / 'logs' / 'a1' / f'train_{run}' / checkpoint.parent.name
    if not log_dir.is_dir():
        print(f'warn {run}: no log folder {log_dir}')
        return {}

    series = {}
    with extract_run(log_dir, 'metrics.csv', None).open(newline='') as handle:
        for row in csv.DictReader(handle):
            series.setdefault(row['tag'], []).append((int(row['step']), float(row['value'])))
    for values in series.values():
        values.sort()
    return series


def scalar_stats(series):
    '''Val accuracy at the epoch BestCheckpoint kept (highest; ties go to the earliest).'''
    accuracy = series.get('valid/accuracy', [])
    best = max(accuracy, key=lambda item: (item[1], -item[0])) if accuracy else (None, None)
    epoch_time = [value for _, value in series.get('training/epoch_time', [])]
    peak = [value for _, value in series.get('training/peak_mem_gb', [])]
    return {
        'best_epoch': best[0],
        'val_acc': best[1],
        'time_per_epoch_s': statistics.mean(epoch_time) if epoch_time else None,
        'train_time_min': sum(epoch_time) / 60 if epoch_time else None,
        'peak_mem_gb': max(peak) if peak else None,
    }


def count_params(info):
    net = timm.create_model(info['timm'], pretrained=False, num_classes=101)
    freeze_parameters(net, info['timm'], info['freeze'])
    total = sum(param.numel() for param in net.parameters())
    trainable = sum(param.numel() for param in net.parameters() if param.requires_grad)
    return total, trainable


def throughput(evaluator, warmup=5, repeats=30):
    '''Inference images/s of the model alone on one test batch kept on the device.'''
    images = next(iter(evaluator.data))[0].to(evaluator.device)
    images = evaluator.to_input(images)
    evaluator.model.eval()
    with torch.no_grad(), evaluator.autocast():
        for _ in range(warmup):
            evaluator.model(images)
        if evaluator.device.type == 'cuda':
            torch.cuda.synchronize(evaluator.device)
        start = time.perf_counter()
        for _ in range(repeats):
            evaluator.model(images)
        if evaluator.device.type == 'cuda':
            torch.cuda.synchronize(evaluator.device)
    return repeats * images.shape[0] / (time.perf_counter() - start)


def evaluate(run, checkpoint, out_dir):
    cfg = CfgNode.load(_repo_root / 'configs' / 'a1' / f'test_{run}.yml')
    cfg.loader.path = str(checkpoint)
    cfg.loader.attach(cfg.evaluator, cfg)
    for name, metric in cfg.metrics.items():
        metric.attach(cfg.evaluator, name)
    matrix = ConfusionMatrix(101)
    matrix.attach(cfg.evaluator, 'confusion_matrix')

    predictions = []

    def keep(engine):
        probs = engine.output['y_pred'].softmax(dim=1)
        confidence, pred = probs.max(dim=1)
        predictions.extend(zip(engine.output['y'].tolist(), pred.tolist(), confidence.tolist()))

    cfg.evaluator.add_event_handler(Events.ITERATION_COMPLETED, keep)
    result = cfg.evaluator.run()
    result['images_per_s'] = throughput(cfg.evaluator)

    dataset = cfg.evaluator.data.dataset
    path = out_dir / 'predictions' / f'{run}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'label', 'pred', 'confidence'])
        for image_id, (label, pred, confidence) in zip(dataset.ids, predictions):
            writer.writerow([image_id, label, pred, f'{confidence:.4f}'])

    return result, dataset


def _fmt(value, digits=4):
    if value is None:
        return ''
    if isinstance(value, int):
        return str(value)
    return f'{value:.{digits}f}'


def write_tables(records, out_dir):
    with (out_dir / 'results_a1.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _fmt(record.get(key)) if isinstance(record.get(key), float)
                             else record.get(key) for key in COLUMNS})

    def cell(value, digits=1, scale=1.0):
        if value is None:
            return '--'
        if isinstance(value, int):
            return f'{value / 1e6:.2f}M'
        return f'{value * scale:.{digits}f}'

    def interval_cell(record):
        low, high = record['test_ci_low'], record['test_ci_high']
        return '--' if low is None else f'[{100 * low:.1f}, {100 * high:.1f}]'

    core = [record for record in records if record['set'] == 'core']
    lines = [
        r'\begin{tabular}{llrrrrrrrr}',
        r'\toprule',
        r'Model & Mode & Trainable & Val acc (\%) & Test acc (\%) & KTC 95\% & Test F1 (\%) & s/epoch & GB & img/s \\',
        r'\midrule',
    ]
    for record in sorted(core, key=lambda item: (item['model'], MODES.index(item['mode']))):
        lines.append(' & '.join((
            record['model'].replace('_', r'\_'),
            record['mode'],
            cell(record['trainable']),
            cell(record['val_acc'], scale=100),
            cell(record['test_acc'], scale=100),
            interval_cell(record),
            cell(record['test_f1'], scale=100),
            cell(record['time_per_epoch_s'], digits=0),
            cell(record['peak_mem_gb']),
            cell(record['images_per_s'], digits=0),
        )) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    (out_dir / 'results_a1.tex').write_text('\n'.join(lines))


def load_correct(run, out_dir):
    '''(ids, boolean array) of the per-image predictions evaluate() wrote for this run.'''
    ids, correct = [], []
    with (out_dir / 'predictions' / f'{run}.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            ids.append(row['id'])
            correct.append(row['label'] == row['pred'])
    return ids, np.array(correct, dtype=bool)


def write_per_class(records, classes, out_dir):
    '''Test accuracy of every class (rows) for every run (columns).'''
    columns = {}
    for record in records:
        matrix = record['result']['confusion_matrix'].cpu().numpy().astype(float)
        totals = matrix.sum(axis=1)
        columns[record['run']] = np.divide(np.diag(matrix), totals, out=np.zeros_like(totals), where=totals > 0)

    with (out_dir / 'per_class.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['class', *columns])
        for index in range(len(next(iter(columns.values())))):
            writer.writerow([_class_name(classes, index), *(f'{values[index]:.4f}' for values in columns.values())])


def write_significance(records, out_dir):
    '''Paired comparisons on the planned contrasts; every run must share the same test images.'''
    correct, reference = {}, None
    for record in records:
        ids, vector = load_correct(record['run'], out_dir)
        if reference is None:
            reference = ids
        elif ids != reference:
            raise SystemExit(f"{record['run']} was scored on different test images than {records[0]['run']}")
        correct[record['run']] = vector

    rows = compare(correct)
    if not rows:
        print('skip significance: no planned comparison has all its runs')
        return

    write_comparisons(rows, out_dir / 'significance.csv', out_dir / 'significance.tex')
    for row in rows:
        print(f"{row['comparison']:55} {row['diff_pts']:+6.2f} pts "
              f"[{row['ci_low_pts']:+.2f}, {row['ci_high_pts']:+.2f}] {row['verdict']}")


def plot_curves(records, out_dir):
    core = [record for record in records if record['set'] == 'core' and record['series']]
    models = sorted({record['model'] for record in core})
    if not models:
        return
    figure, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.4), squeeze=False)
    for axis, model in zip(axes[0], models):
        for record in core:
            if record['model'] != model:
                continue
            points = record['series'].get('valid/accuracy', [])
            axis.plot([step for step, _ in points], [value for _, value in points], label=record['mode'])
        axis.set_title(model)
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Val accuracy')
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / 'curves_a1.png', dpi=150)
    plt.close(figure)


def plot_cost(records, out_dir):
    points = [record for record in records if record['set'] == 'core'
              and record['train_time_min'] is not None and record['test_acc'] is not None]
    if not points:
        return
    figure, axis = plt.subplots(figsize=(6, 3.8))
    for record in points:
        marker = 'o' if record['model'] == 'resnet50' else 's'
        axis.scatter(record['train_time_min'], 100 * record['test_acc'], marker=marker, s=45,
                     label=f"{record['model']}/{record['mode']}")
    axis.set_xlabel('Training time (min)')
    axis.set_ylabel('Test accuracy (%)')
    axis.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(out_dir / 'cost_a1.png', dpi=150)
    plt.close(figure)


def _class_name(classes, index):
    return classes[index] if index < len(classes) else str(index)


def plot_confused_pairs(record, classes, out_dir, top=15):
    counts = record['result']['confusion_matrix'].cpu().numpy().copy()
    np.fill_diagonal(counts, 0)
    order = np.argsort(counts, axis=None)[::-1][:top]
    pairs = [np.unravel_index(index, counts.shape) for index in order]
    pairs = [(t, p) for t, p in pairs if counts[t, p] > 0]
    if not pairs:
        return
    figure, axis = plt.subplots(figsize=(6, 4.2))
    labels = [f'{_class_name(classes, t)} → {_class_name(classes, p)}' for t, p in pairs]
    axis.barh(range(len(pairs)), [counts[t, p] for t, p in pairs])
    axis.set_yticks(range(len(pairs)), labels, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlabel('Test images')
    axis.set_title(f"{record['run']}: most confused pairs (true → predicted)", fontsize=9)
    figure.tight_layout()
    figure.savefig(out_dir / f"confused_{record['run']}.png", dpi=150)
    plt.close(figure)


def plot_errors(record, dataset, out_dir, rows=3, cols=6):
    '''The most confident mistakes: label noise, look-alike dishes, or real model errors?'''
    path = out_dir / 'predictions' / f"{record['run']}.csv"
    with path.open(newline='') as handle:
        wrong = [row for row in csv.DictReader(handle) if row['label'] != row['pred']]
    wrong.sort(key=lambda row: -float(row['confidence']))
    shown = wrong[:rows * cols]
    if not shown:
        return
    figure, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.3 * rows), squeeze=False)
    for axis, row in zip(axes.flat, shown):
        with Image.open(dataset.images_folder / f"{row['id']}.jpg") as image:
            axis.imshow(image.convert('RGB'))
        true, pred = _class_name(dataset.classes, int(row['label'])), _class_name(dataset.classes, int(row['pred']))
        axis.set_title(f'{true}\n→ {pred} ({float(row["confidence"]):.2f})', fontsize=7)
    for axis in axes.flat:
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(out_dir / f"errors_{record['run']}.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Compare the A1 runs.')
    parser.add_argument('--runs', default=None, help='a set from runs.json or comma-separated run names')
    parser.add_argument('--out-dir', default=str(_repo_root / 'docs' / 'assets' / 'a1'))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    dataset = None
    for run, info in load_runs(args.runs).items():
        checkpoint = latest_best(run)
        if checkpoint is None:
            print(f'skip {run}: no best_*.pt')
            continue
        print(f'analyze {run} <- {checkpoint.relative_to(_repo_root)}')
        result, dataset = evaluate(run, checkpoint, out_dir)
        series = read_series(run, checkpoint)
        total, trainable = count_params(info)
        matrix = result['confusion_matrix']
        low, high = wilson_interval(int(matrix.trace()), int(matrix.sum()))
        records.append({
            **info,
            'run': run,
            'params': total,
            'trainable': trainable,
            'test_acc': result.get('accuracy'),
            'test_ci_low': low,
            'test_ci_high': high,
            'test_f1': result.get('f1_score'),
            'images_per_s': result.get('images_per_s'),
            'result': result,
            'series': series,
            **scalar_stats(series),
        })

    if not records:
        raise SystemExit('no finished runs to analyze')

    write_tables(records, out_dir)
    write_per_class(records, dataset.classes, out_dir)
    write_significance(records, out_dir)
    plot_curves(records, out_dir)
    plot_cost(records, out_dir)

    # Error analysis on the best CNN and the best Transformer (chosen on val, not test).
    for model in sorted({record['model'] for record in records}):
        candidates = [record for record in records if record['model'] == model and record['val_acc'] is not None]
        if candidates:
            best = max(candidates, key=lambda record: record['val_acc'])
            plot_confused_pairs(best, dataset.classes, out_dir)
            plot_errors(best, dataset, out_dir)

    print(f'wrote results, significance, per-class tables and figures to {out_dir}')


if __name__ == '__main__':
    main()
