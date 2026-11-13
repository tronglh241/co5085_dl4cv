'''Paired bootstrap confidence intervals for the A2 test verification results.

Reads the per-pair test scores written by evaluate.py (reports/results/a2/scores/<run>_test.csv) and
resamples the 10,000 test pairs with replacement. The same resample is applied to every run, so the
difference between two runs is paired. Only verification can be done here: the identification
embeddings are not saved.
'''

import argparse
import csv
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

RUNS = ('resnet50_softmax', 'resnet50_arcface', 'mobilenetv3_softmax', 'mobilenetv3_arcface')
# (a, b): the difference reported is metric(a) - metric(b).
PAIRS = (
    ('resnet50_arcface', 'resnet50_softmax'),
    ('mobilenetv3_arcface', 'mobilenetv3_softmax'),
    ('mobilenetv3_arcface', 'resnet50_arcface'),
    ('mobilenetv3_softmax', 'resnet50_softmax'),
    ('resnet50_arcface', 'resnet50_softmax_lr0.01'),
)


def read_scores(path):
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    return np.array([int(row['label']) for row in rows]), np.array([float(row['score']) for row in rows])


def auc(labels, scores):
    return float(roc_auc_score(labels, scores))


def eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


METRICS = {'roc_auc': auc, 'eer': eer}


def interval(values):
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


# Label, scale and decimals of each metric in the table: the EER difference is shown in percentage points.
TEX_METRICS = {'roc_auc': ('ROC-AUC', 1.0, 4), 'eer': (r'EER (điểm \%)', 100.0, 2)}


def write_tex(rows, path):
    lines = [
        r'\begin{tabular}{llrrr}',
        r'\toprule',
        r'Phép so sánh & Metric & Chênh lệch & \multicolumn{2}{c}{KTC $95\%$} \\',
        r'\midrule',
    ]
    for row in rows:
        name = row['comparison'].replace('_', r'\_')
        label, scale, digits = TEX_METRICS[row['metric']]
        values = ' & '.join(f'{row[key] * scale:+.{digits}f}' for key in ('diff', 'low', 'high'))
        lines.append(f'{name} & {label} & {values} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Bootstrap intervals for the A2 test scores.')
    parser.add_argument('--scores-dir', default='reports/results/a2/scores')
    parser.add_argument('--out-dir', default='reports/results/a2')
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir)
    data = {}
    for path in sorted(scores_dir.glob('*_test.csv')):
        data[path.name.removesuffix('_test.csv')] = read_scores(path)

    missing = [run for run in RUNS if run not in data]
    if missing:
        raise SystemExit(f'no test scores for {missing} in {scores_dir}')

    reference = data[RUNS[0]][0]
    for run, (labels, _) in data.items():
        if len(labels) != len(reference) or not np.array_equal(labels, reference):
            raise SystemExit(f'{run}: the pairs differ from those of {RUNS[0]}, a paired bootstrap is not valid')

    rng = np.random.default_rng(args.seed)
    count = len(reference)
    resamples = [rng.integers(0, count, size=count) for _ in range(args.resamples)]

    # Metric of every run on every resample, computed once.
    boot = {run: {name: np.empty(args.resamples) for name in METRICS} for run in data}
    point = {}
    for run, (labels, scores) in data.items():
        point[run] = {name: function(labels, scores) for name, function in METRICS.items()}
        for index, picks in enumerate(resamples):
            for name, function in METRICS.items():
                boot[run][name][index] = function(labels[picks], scores[picks])

    per_run = []
    for run in data:
        for name in METRICS:
            low, high = interval(boot[run][name])
            per_run.append({'run': run, 'metric': name, 'value': point[run][name], 'low': low, 'high': high})

    rows = []
    for first, second in PAIRS:
        if first not in data or second not in data:
            continue
        for name in METRICS:
            difference = boot[first][name] - boot[second][name]
            low, high = interval(difference)
            rows.append({
                'comparison': f'{first} - {second}',
                'metric': name,
                'diff': point[first][name] - point[second][name],
                'low': low,
                'high': high,
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'bootstrap_runs_a2.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['run', 'metric', 'value', 'low', 'high'])
        writer.writeheader()
        writer.writerows(per_run)
    with (out_dir / 'bootstrap_a2.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['comparison', 'metric', 'diff', 'low', 'high'])
        writer.writeheader()
        writer.writerows(rows)
    write_tex(rows, out_dir / 'bootstrap_a2.tex')

    print(f'{args.resamples} resamples of {count} pairs, seed {args.seed}')
    for row in per_run:
        print(f"{row['run']:34s} {row['metric']:8s} {row['value']:.4f}  [{row['low']:.4f}, {row['high']:.4f}]")
    for row in rows:
        verdict = 'excludes 0' if row['low'] > 0 or row['high'] < 0 else 'includes 0'
        print(f"{row['comparison']:58s} {row['metric']:8s} {row['diff']:+.4f}  "
              f"[{row['low']:+.4f}, {row['high']:+.4f}] {verdict}")


if __name__ == '__main__':
    main()
