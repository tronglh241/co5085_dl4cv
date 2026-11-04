'''Build A2 tables and figures from evaluated checkpoints.'''

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from scripts.tensorboard.extract import extract_run

RUNS_JSON = _repo_root / 'configs' / 'a2' / 'runs.json'
COLUMNS = (
    'run', 'model', 'loss', 'params', 'best_epoch', 'val_roc_auc', 'val_eer',
    'test_roc_auc', 'test_eer', 'test_acc', 'test_threshold', 'tar_far_1e-3',
    'tar_far_1e-2', 'rank1', 'recall5', 'map', 'time_per_epoch_s',
    'train_time_min', 'peak_mem_gb',
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


def read_series(run, result):
    checkpoint = result.get('checkpoint')
    if checkpoint:
        log_dir = _repo_root / 'logs' / 'a2' / f'train_{run}' / Path(checkpoint).parent.name
    else:
        log_dir = None

    if log_dir is None or not log_dir.is_dir():
        found = sorted((_repo_root / 'logs' / 'a2' / f'train_{run}').glob('*/'))
        log_dir = found[-1] if found else None

    if log_dir is None:
        print(f'warn {run}: no log folder')
        return {}

    try:
        metrics = extract_run(log_dir, 'metrics.csv', None)
    except (ValueError, KeyError, RuntimeError) as error:
        print(f'warn {run}: cannot read {log_dir}: {error}')
        return {}

    series = {}
    with metrics.open(newline='') as handle:
        for row in csv.DictReader(handle):
            series.setdefault(row['tag'], []).append((int(row['step']), float(row['value'])))

    for values in series.values():
        values.sort()
    return series


def series_stats(series):
    roc_auc = series.get('valid/roc_auc', [])
    best = max(roc_auc, key=lambda item: (item[1], -item[0])) if roc_auc else (None, None)
    epoch_time = [value for _, value in series.get('training/epoch_time', [])]
    peak = [value for _, value in series.get('training/peak_mem_gb', [])]
    return {
        'best_epoch': best[0],
        'val_roc_auc_from_log': best[1],
        'time_per_epoch_s': statistics.mean(epoch_time) if epoch_time else None,
        'train_time_min': sum(epoch_time) / 60 if epoch_time else None,
        'peak_mem_gb': max(peak) if peak else None,
    }


def make_record(run, info, result):
    val, test, ident = result['val'], result['test'], result.get('identification') or {}
    series = read_series(run, result)
    stats = series_stats(series)
    return {
        'run': run,
        'model': info['model'],
        'loss': info['loss'],
        'params': result.get('params'),
        'best_epoch': stats['best_epoch'],
        'val_roc_auc': val.get('roc_auc'),
        'val_eer': val.get('eer'),
        'test_roc_auc': test.get('roc_auc'),
        'test_eer': test.get('eer'),
        'test_acc': test.get('accuracy'),
        'test_threshold': test.get('threshold'),
        'tar_far_1e-3': test.get('tar_at_far_1e-3'),
        'tar_far_1e-2': test.get('tar_at_far_1e-2'),
        'rank1': ident.get('rank1'),
        'recall5': ident.get('recall5'),
        'map': ident.get('map'),
        'time_per_epoch_s': stats['time_per_epoch_s'],
        'train_time_min': stats['train_time_min'],
        'peak_mem_gb': stats['peak_mem_gb'],
        'series': series,
    }


def _fmt(value, digits=4):
    if value is None:
        return ''
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def write_csv(records, results_dir):
    path = results_dir / 'results_a2.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _fmt(record.get(key)) for key in COLUMNS})
    print(f'wrote {path}')


def _tex(text):
    return text.replace('_', r'\_')


def _millions(value):
    return '--' if value is None else f'{value / 1e6:.2f}M'


def _percent(value, digits=2):
    return '--' if value is None else f'{100 * value:.{digits}f}'


def write_tables(records, results_dir):
    lines = [
        r'\begin{tabular}{llrrrrrrrr}',
        r'\toprule',
        r'Run & Loss & Params & Val ROC-AUC & Test ROC-AUC & EER (\%) & Acc. (\%) '
        r'& TAR@FAR $10^{-3}$ & TAR@FAR $10^{-2}$ & Rank-1 (\%) \\',
        r'\midrule',
    ]
    for record in records:
        lines.append(' & '.join((
            _tex(record['run']),
            record['loss'],
            _millions(record['params']),
            _percent(record['val_roc_auc'], 3),
            _percent(record['test_roc_auc'], 3),
            _percent(record['test_eer']),
            _percent(record['test_acc']),
            _fmt(record['tar_far_1e-3'], 3),
            _fmt(record['tar_far_1e-2'], 3),
            _percent(record['rank1']),
        )) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    path = results_dir / 'results_a2.tex'
    path.write_text('\n'.join(lines))
    print(f'wrote {path}')

    ident = [
        r'\begin{tabular}{lrrr}',
        r'\toprule',
        r'Run & Rank-1 (\%) & Recall@5 (\%) & mAP (\%) \\',
        r'\midrule',
    ]
    for record in records:
        ident.append(' & '.join((
            _tex(record['run']),
            _percent(record['rank1']),
            _percent(record['recall5']),
            _percent(record['map']),
        )) + r' \\')
    ident += [r'\bottomrule', r'\end{tabular}', '']
    (results_dir / 'identification_a2.tex').write_text('\n'.join(ident))


def plot_roc(records, scores_dir, figures_dir):
    drawn = False
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    for record in records:
        path = scores_dir / f"{record['run']}_test.csv"
        if not path.exists():
            continue
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        labels = np.array([int(row['label']) for row in rows])
        scores = np.array([float(row['score']) for row in rows])
        fpr, tpr, _ = roc_curve(labels, scores)
        axis.plot(fpr, tpr, label=f"{record['run']} ({roc_auc_score(labels, scores):.4f})")
        drawn = True

    if not drawn:
        plt.close(figure)
        return

    axis.plot([0, 1], [0, 1], linestyle='--', color='0.6', linewidth=0.8)
    axis.set_xlabel('False positive rate')
    axis.set_ylabel('True positive rate')
    axis.set_title('CelebA test verification')
    axis.legend(fontsize=7, loc='lower right')
    figure.tight_layout()
    figure.savefig(figures_dir / 'roc_a2.png', dpi=300)
    plt.close(figure)


def plot_bars(records, figures_dir):
    with_values = [record for record in records if record['test_roc_auc'] is not None]
    if not with_values:
        return

    names = [record['run'] for record in with_values]
    figure, axes = plt.subplots(1, 2, figsize=(10, 0.5 * len(names) + 2.5))
    for axis, key, title in (
        (axes[0], 'test_roc_auc', 'Test ROC-AUC'),
        (axes[1], 'rank1', 'Test Rank-1'),
    ):
        values = [record[key] for record in with_values]
        axis.barh(names, values, color='C0')
        axis.set_xlim(0, 1)
        axis.set_title(title)
        axis.invert_yaxis()
        axis.grid(axis='x', alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures_dir / 'bars_a2.png', dpi=300)
    plt.close(figure)


def plot_curves(records, figures_dir):
    drawn = False
    figure, axis = plt.subplots(figsize=(6, 3.8))
    for record in records:
        points = record['series'].get('valid/roc_auc', [])
        if not points:
            continue
        axis.plot([step for step, _ in points], [value for _, value in points], label=record['run'])
        drawn = True

    if not drawn:
        plt.close(figure)
        return

    axis.set_xlabel('Epoch')
    axis.set_ylabel('Validation ROC-AUC')
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(figures_dir / 'curves_a2.png', dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Build A2 tables and figures.')
    parser.add_argument('--runs', default=None, help='run name, set name, or comma-separated runs')
    parser.add_argument('--results-dir', default='reports/results/a2')
    parser.add_argument('--figures-dir', default='docs/assets/a2')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    scores_dir = results_dir / 'scores'
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for run, info in load_runs(args.runs).items():
        path = results_dir / f'{run}.json'
        if not path.exists():
            print(f'skip {run}: {path} is missing')
            continue
        records.append(make_record(run, info, json.loads(path.read_text())))

    if not records:
        print('no evaluated A2 runs yet')
        return

    write_csv(records, results_dir)
    write_tables(records, results_dir)
    plot_roc(records, scores_dir, figures_dir)
    plot_bars(records, figures_dir)
    plot_curves(records, figures_dir)


if __name__ == '__main__':
    main()
