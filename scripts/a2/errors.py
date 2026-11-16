'''Failure-case analysis of the A2 verification results.

Uses the per-pair test scores written by evaluate.py (reports/results/a2/scores/<run>_test.csv, in the order of
data/celeba_112/pairs/test.csv) and the validation threshold stored in each run's result file, so it needs no
model and no GPU. It reports, for each run:
  1. false accepts (different people judged the same) and false rejects (same person judged different);
  2. how many errors are shared between runs, to separate "hard pairs" from "model-specific mistakes";
  3. error rate against simple image-quality proxies (sharpness, brightness) of the worse image in the pair;
and saves the most confident mistakes of the chosen run as an image grid.

Quality is measured on the 112 x 112 aligned crop, so it describes what the network sees. CelebA attributes
(pose, glasses, age) are not part of the prepared data, so they are not analysed here.
'''

import argparse
import csv
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.a2.result_io import read_result  # noqa: E402

RUNS = ('resnet50_softmax', 'resnet50_arcface', 'mobilenetv3_softmax', 'mobilenetv3_arcface')
LABELS = {
    'resnet50_softmax': 'ResNet-50 softmax',
    'resnet50_arcface': 'ResNet-50 ArcFace',
    'mobilenetv3_softmax': 'MobileNetV3 softmax',
    'mobilenetv3_arcface': 'MobileNetV3 ArcFace',
}


def read_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def sharpness(gray):
    '''Variance of a 4-neighbour Laplacian: low for blurred images.'''
    g = gray.astype(np.float64)
    laplacian = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(laplacian.var())


def image_quality(folder, names):
    '''Sharpness and mean brightness (0 to 255) of each image in names.'''
    table = {}
    for name in names:
        with Image.open(Path(folder) / name) as image:
            gray = np.asarray(image.convert('L'))
        table[name] = (sharpness(gray), float(gray.mean()))
    return table


def error_rate_by_tercile(values, wrong, mask):
    '''Error rate of the pairs in mask, split into three equally sized groups by values.'''
    values, wrong = values[mask], wrong[mask]
    edges = np.quantile(values, [1 / 3, 2 / 3])
    groups = np.digitize(values, edges)
    return [float(wrong[groups == g].mean()) for g in range(3)], [int((groups == g).sum()) for g in range(3)]


def contact_sheet(path, rows, folder, title, columns=6):
    '''rows: list of (name_a, name_b, score). Saves pairs of aligned faces with their cosine score.'''
    count = min(len(rows), columns * 2)
    cols = columns
    grid_rows = (count + cols - 1) // cols
    figure, axes = plt.subplots(grid_rows, cols, figsize=(1.9 * cols, 1.25 * grid_rows + 0.5))
    axes = np.atleast_2d(axes)
    for axis in axes.ravel():
        axis.set_axis_off()
    for axis, (name_a, name_b, score) in zip(axes.ravel(), rows[:count]):
        with Image.open(Path(folder) / name_a) as a, Image.open(Path(folder) / name_b) as b:
            pair = np.concatenate([np.asarray(a.convert('RGB')), np.asarray(b.convert('RGB'))], axis=1)
        axis.imshow(pair)
        axis.set_title(f'{score:.2f}', fontsize=8, pad=2)
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Failure-case analysis of the A2 test pairs.')
    parser.add_argument('--root', default='data/celeba_112')
    parser.add_argument('--results-dir', default='reports/results/a2')
    parser.add_argument('--assets-dir', default='docs/assets/a2')
    parser.add_argument('--run', default='mobilenetv3_arcface', choices=RUNS, help='run shown in the figures')
    parser.add_argument('--examples', type=int, default=12)
    args = parser.parse_args()

    root, results, assets = Path(args.root), Path(args.results_dir), Path(args.assets_dir)
    pairs = read_rows(root / 'pairs' / 'test.csv')
    labels = np.array([int(row['label']) for row in pairs])
    positives, negatives = labels == 1, labels == 0

    scores, wrong, thresholds = {}, {}, {}
    for run in RUNS:
        rows = read_rows(results / 'scores' / f'{run}_test.csv')
        if len(rows) != len(pairs) or any(int(r['label']) != int(p['label']) for r, p in zip(rows, pairs)):
            raise SystemExit(f'{run}: scores do not line up with pairs/test.csv')
        thresholds[run] = read_result(results / f'{run}.txt')['val']['threshold']
        scores[run] = np.array([float(row['score']) for row in rows])
        wrong[run] = (scores[run] >= thresholds[run]) != (labels == 1)

    # 1. false accept / false reject
    summary = []
    for run in RUNS:
        fa = int((wrong[run] & negatives).sum())
        fr = int((wrong[run] & positives).sum())
        summary.append({
            'run': run, 'threshold': thresholds[run],
            'false_accepts': fa, 'far': fa / int(negatives.sum()),
            'false_rejects': fr, 'frr': fr / int(positives.sum()),
            'errors': fa + fr, 'accuracy': 1.0 - (fa + fr) / len(labels),
        })

    # 2. overlap between runs
    arc = {'resnet50': 'resnet50_arcface', 'mobilenetv3': 'mobilenetv3_arcface'}
    soft = {'resnet50': 'resnet50_softmax', 'mobilenetv3': 'mobilenetv3_softmax'}
    overlap = []
    for backbone in arc:
        a, s = wrong[arc[backbone]], wrong[soft[backbone]]
        overlap.append({
            'pair': f'{backbone}: arcface vs softmax', 'arcface_errors': int(a.sum()), 'softmax_errors': int(s.sum()),
            'both': int((a & s).sum()), 'share_of_arcface': float((a & s).sum() / a.sum()),
        })
    a, b = wrong['resnet50_arcface'], wrong['mobilenetv3_arcface']
    overlap.append({
        'pair': 'arcface: resnet50 vs mobilenetv3', 'arcface_errors': int(a.sum()), 'softmax_errors': int(b.sum()),
        'both': int((a & b).sum()), 'share_of_arcface': float((a & b).sum() / a.sum()),
    })
    all_four = np.logical_and.reduce([wrong[run] for run in RUNS])
    any_one = np.logical_or.reduce([wrong[run] for run in RUNS])

    # 3. quality proxies on the worse image of each pair
    names = sorted({row[key] for row in pairs for key in ('id_a', 'id_b')})
    quality = image_quality(root / 'images', names)
    sharp = np.array([min(quality[r['id_a']][0], quality[r['id_b']][0]) for r in pairs])
    dark = np.array([min(quality[r['id_a']][1], quality[r['id_b']][1]) for r in pairs])
    bright = np.array([max(quality[r['id_a']][1], quality[r['id_b']][1]) for r in pairs])
    contrast_gap = np.array([abs(quality[r['id_a']][1] - quality[r['id_b']][1]) for r in pairs])

    factors = (
        ('sharpness of the blurrier image', sharp),
        ('brightness of the darker image', dark),
        ('brightness of the brighter image', bright),
        ('brightness gap between the two images', contrast_gap),
    )
    quality_rows = []
    for run in RUNS:
        for factor, values in factors:
            for kind, mask in (('same person (false reject)', positives), ('different people (false accept)', negatives)):
                rates, sizes = error_rate_by_tercile(values, wrong[run], mask)
                quality_rows.append({'run': run, 'factor': factor, 'kind': kind,
                                     'low': rates[0], 'mid': rates[1], 'high': rates[2], 'n_per_group': sizes[0]})

    # outputs
    results.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    with (results / 'errors_a2.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (results / 'errors_quality_a2.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quality_rows[0]))
        writer.writeheader()
        writer.writerows(quality_rows)

    print('run                    threshold  false_acc  FAR      false_rej  FRR      errors  accuracy')
    for row in summary:
        print(f"{row['run']:22s} {row['threshold']:8.4f}  {row['false_accepts']:8d}  {row['far']:.4f}  "
              f"{row['false_rejects']:9d}  {row['frr']:.4f}  {row['errors']:6d}  {row['accuracy']:.4f}")
    print(f'\nnegative pairs {int(negatives.sum())}, positive pairs {int(positives.sum())}')
    print('\noverlap of errors:')
    for row in overlap:
        print(f"  {row['pair']:36s} {row['arcface_errors']:5d} / {row['softmax_errors']:5d}  both {row['both']:5d}  "
              f"({row['share_of_arcface']:.1%} of the first)")
    print(f'  wrong in all four runs: {int(all_four.sum())}; wrong in at least one: {int(any_one.sum())}')
    print('\nerror rate by tercile (low / mid / high value of the factor):')
    for row in quality_rows:
        if row['run'] == args.run:
            print(f"  {row['factor']:40s} {row['kind']:32s} {row['low']:.3f} {row['mid']:.3f} {row['high']:.3f}"
                  f"  (n={row['n_per_group']})")

    # figures for the chosen run
    run = args.run
    folder = root / 'images'
    fr_idx = np.where(wrong[run] & positives)[0]
    fa_idx = np.where(wrong[run] & negatives)[0]
    fr_idx = fr_idx[np.argsort(scores[run][fr_idx])][:args.examples]
    fa_idx = fa_idx[np.argsort(-scores[run][fa_idx])][:args.examples]
    contact_sheet(assets / 'errors_false_reject_a2.png',
                  [(pairs[i]['id_a'], pairs[i]['id_b'], scores[run][i]) for i in fr_idx], folder,
                  f'False rejects (same person, lowest cosine), {LABELS[run]}, threshold {thresholds[run]:.2f}')
    contact_sheet(assets / 'errors_false_accept_a2.png',
                  [(pairs[i]['id_a'], pairs[i]['id_b'], scores[run][i]) for i in fa_idx], folder,
                  f'False accepts (different people, highest cosine), {LABELS[run]}, threshold {thresholds[run]:.2f}')

    # error rate by sharpness for the four runs
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)
    for axis, (mask, title) in zip(axes, ((positives, 'Same person: false reject rate'),
                                           (negatives, 'Different people: false accept rate'))):
        for r in RUNS:
            rates, _ = error_rate_by_tercile(sharp, wrong[r], mask)
            axis.plot(['blurriest third', 'middle', 'sharpest third'], rates, marker='o', label=LABELS[r])
        axis.set_title(title, fontsize=9)
        axis.set_ylabel('error rate')
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(assets / 'errors_sharpness_a2.png', dpi=200)
    plt.close(figure)


if __name__ == '__main__':
    main()
