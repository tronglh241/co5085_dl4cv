'''Uncertainty and paired comparisons of the A1 test results.

Every run is scored on the same 25,250 test images, and analyze.py keeps each
run's per-image predictions. That allows three things beyond a bare accuracy:

- a Wilson interval for each run's test accuracy,
- a paired bootstrap interval for the difference between two runs, or between
  two such differences (resample test images, score every run on the same
  resample),
- an exact McNemar test for a pair of runs, which looks only at the images on
  which exactly one of the two is right.

All three measure the sampling error of the test set. None of them measures
the run-to-run variation of training, which needs several seeds; every run
here has one. The verdict column therefore also demands a minimum effect size.
'''

import csv
import math

import numpy as np

Z95 = 1.959963984540054
# A difference is called clear only if its interval excludes 0 and it is at
# least this many accuracy points; smaller gaps are within what another seed
# could change.
MIN_EFFECT_PTS = 1.0
MODES = ('scratch', 'head', 'partial', 'full')
MODELS = ('resnet50', 'deit_s')


def wilson_interval(correct, total, z=Z95):
    '''Wilson score interval (low, high) of a proportion, as fractions.'''
    if total == 0:
        return None, None

    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return centre - half, centre + half


def mcnemar_exact(correct_a, correct_b):
    '''(b, c, p): b images only A gets right, c only B, two-sided exact p-value.

    Under the null the discordant images split evenly between A and B, so the
    smaller count is Binomial(b + c, 1/2).
    '''
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n = only_a + only_b

    if n == 0:
        return only_a, only_b, 1.0

    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return only_a, only_b, min(1.0, 2 * tail / 2 ** n)


def paired_bootstrap(correct, weights, resamples=2000, seed=0, chunk=100):
    '''Estimate and 95% interval of sum(weight * accuracy) over runs, in accuracy units.

    correct maps run -> boolean array over the same test images; weights maps
    run -> coefficient. {A: 1, B: -1} is accuracy(A) - accuracy(B); four runs
    with +1, -1, -1, +1 is a difference of differences.
    '''
    runs = list(weights)
    stacked = np.stack([correct[run] for run in runs]).astype(np.float64)
    coefficients = np.array([weights[run] for run in runs])
    n = stacked.shape[1]
    estimate = float(coefficients @ stacked.mean(axis=1))
    rng = np.random.default_rng(seed)
    draws = []

    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        index = rng.integers(0, n, size=(size, n))
        # (runs, size): accuracy of every run on every resample.
        resampled = stacked[:, index].mean(axis=2)
        draws.append(coefficients @ resampled)

    draws = np.concatenate(draws)
    low, high = np.percentile(draws, [2.5, 97.5])
    return estimate, float(low), float(high)


def planned_comparisons():
    '''(group, label, {run: coefficient}) for the comparisons the protocol fixes in advance.

    family    CNN against Transformer in the same mode     (question 1)
    pretrain  scratch -> full finetune, per model, and the gap of the
              Transformer minus the gap of the CNN         (question 2)
    freeze    cost of freezing: head and partial against full (question 3)
    '''
    comparisons = []

    for mode in MODES:
        comparisons.append(('family', f'{mode}: deit_s - resnet50', {f'deit_s_{mode}': 1, f'resnet50_{mode}': -1}))

    for model in MODELS:
        comparisons.append(('pretrain', f'{model}: full - scratch', {f'{model}_full': 1, f'{model}_scratch': -1}))

    comparisons.append((
        'pretrain',
        '(deit_s full - scratch) - (resnet50 full - scratch)',
        {'deit_s_full': 1, 'deit_s_scratch': -1, 'resnet50_full': -1, 'resnet50_scratch': 1},
    ))

    for model in MODELS:
        for mode in ('head', 'partial'):
            comparisons.append(('freeze', f'{model}: {mode} - full', {f'{model}_{mode}': 1, f'{model}_full': -1}))

    return comparisons


def holm(p_values):
    '''Holm-Bonferroni adjusted p-values, in the input order.'''
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0

    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running

    return adjusted


def verdict(low, high, estimate_pts):
    if low > 0 or high < 0:
        return 'clear' if abs(estimate_pts) >= MIN_EFFECT_PTS else 'small'

    return 'inconclusive'


def compare(correct, resamples=2000):
    '''One row per planned comparison whose runs all have predictions.'''
    rows = []

    for group, label, weights in planned_comparisons():
        if not all(run in correct for run in weights):
            continue

        estimate, low, high = paired_bootstrap(correct, weights, resamples=resamples)
        row = {
            'group': group,
            'comparison': label,
            'diff_pts': 100 * estimate,
            'ci_low_pts': 100 * low,
            'ci_high_pts': 100 * high,
            'only_first': None,
            'only_second': None,
            'p': None,
        }

        # McNemar needs exactly two runs: the one with +1 against the one with -1.
        if len(weights) == 2:
            first = next(run for run, weight in weights.items() if weight > 0)
            second = next(run for run, weight in weights.items() if weight < 0)
            row['only_first'], row['only_second'], row['p'] = mcnemar_exact(correct[first], correct[second])

        row['verdict'] = verdict(low, high, 100 * estimate)
        rows.append(row)

    pairs = [row for row in rows if row['p'] is not None]
    for row, adjusted in zip(pairs, holm([row['p'] for row in pairs])):
        row['p_holm'] = adjusted

    return rows


def _csv_value(key, value):
    if value is None:
        return ''

    if key in ('p', 'p_holm'):
        return f'{value:.4g}'

    return f'{value:.2f}' if isinstance(value, float) else value


def write_comparisons(rows, csv_path, tex_path):
    fields = ('group', 'comparison', 'diff_pts', 'ci_low_pts', 'ci_high_pts',
              'only_first', 'only_second', 'p', 'p_holm', 'verdict')

    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(key, row.get(key)) for key in fields})

    def p_cell(value):
        if value is None:
            return '--'
        return '<0.001' if value < 0.001 else f'{value:.3f}'

    lines = [
        r'\begin{tabular}{lrrrl}',
        r'\toprule',
        r'So sánh & Chênh (điểm) & KTC 95\% & \(p\) Holm & Kết luận \\',
        r'\midrule',
    ]
    previous = None
    for row in rows:
        if previous is not None and row['group'] != previous:
            lines.append(r'\addlinespace')
        previous = row['group']
        lines.append(' & '.join((
            row['comparison'].replace('_', r'\_'),
            f"{row['diff_pts']:+.2f}",
            f"[{row['ci_low_pts']:+.2f}, {row['ci_high_pts']:+.2f}]",
            p_cell(row.get('p_holm')),
            row['verdict'],
        )) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    tex_path.write_text('\n'.join(lines))
