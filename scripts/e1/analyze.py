import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def _display_path(path):
    path = Path(path)
    return str(path.relative_to(_repo_root)) if path.is_relative_to(_repo_root) else str(path)


def _architecture_text(record, model, checkpoint):
    lines = [
        f"run: {record['run']}",
        f"model: {type(model).__module__}.{type(model).__name__}",
        f"variant: {record['variant']}",
        f"params: {record['params']}",
        f"checkpoint: {_display_path(checkpoint)}",
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
    stats = _epoch_stats(_load_series(run, checkpoint))
    result = _evaluate(cfg)
    model = cfg.model

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


def _plot_confusion(matrix, run, figures_dir):
    counts = matrix.cpu().numpy().astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    rates = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0) * 100
    errors = rates.copy()
    np.fill_diagonal(errors, np.nan)

    figure, axis = plt.subplots(figsize=(5.4, 4.6))
    image = axis.imshow(errors, cmap='Reds', vmin=0)
    figure.colorbar(image, ax=axis, label='% of true class')

    finite = errors[np.isfinite(errors)]
    limit = finite.max() if finite.size else 0
    labels = range(counts.shape[0])

    for row in labels:
        for col in labels:
            if row != col and counts[row, col] > 0:
                dark = errors[row, col] > limit / 2
                axis.text(col, row, int(counts[row, col]), ha='center', va='center',
                          fontsize=7, color='white' if dark else 'black')

    axis.set_xticks(list(labels))
    axis.set_yticks(list(labels))
    axis.set_yticklabels([f'{row} ({rates[row, row]:.1f}%)' for row in labels], fontsize=8)
    axis.set_xlabel('Predicted')
    axis.set_ylabel('True (per-class accuracy)')
    total_errors = int(counts.sum() - np.trace(counts))
    axis.set_title(f'{run}: {total_errors} errors / {int(counts.sum())}', fontsize=9)
    figure.tight_layout()
    figure.savefig(figures_dir / f'confusion_{run}.png', dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Write E1 metrics, architectures and confusion matrices.')
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
        _plot_confusion(matrix, run, figures_dir)

    print(f'wrote {path}, architectures and confusion matrices for {len(records)} runs')


if __name__ == '__main__':
    main()
