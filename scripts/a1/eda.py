'''Exploratory data analysis of Food-101 for A1.1 and the A1 page.

Writes figures into --out (default docs/assets/a1/eda):
    class_counts.png         images per class in train, val and test
    image_sizes.png          width, height and shorter side of the original images
    samples.png              a few train images of a few classes
    augmentation.png         one image through the shared train pipeline several times

Writes tables and values into --results-dir (default reports/results/a1/eda):
    class_counts.csv         the same counts as class_counts.png
    summary.json / .md       totals, size statistics, image modes, channel mean/std
    splits.tex, stats.tex    the same numbers as booktabs tables for reports/assignment_1.tex

    python scripts/a1/eda.py --raw data/ --root data/food101_256

--raw is the original download (sizes are measured there, before resizing);
--root is the prepared copy the models train on.
'''

import argparse
import csv
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from datasets.food101 import Food101
from scripts.a1.make_configs import EVAL_STEPS, MEAN, STD, TRAIN_STEPS

SPLITS = ('train', 'val', 'test')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def pipeline(steps):
    '''The config steps up to the PIL image, before ToTensor and Normalize.'''
    return eval('transforms.Compose([' + ', '.join(steps) + '])', {'transforms': transforms})


def load_splits(root):
    return {split: Food101(root, split=split) for split in SPLITS}


def class_counts(splits, out, results):
    classes = splits['train'].classes
    counts = {split: Counter(dataset.targets) for split, dataset in splits.items()}

    with (results / 'class_counts.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['class', *SPLITS])
        for index, name in enumerate(classes):
            writer.writerow([name, *(counts[split][index] for split in SPLITS)])

    figure, axis = plt.subplots(figsize=(14, 3.6))
    bottom = np.zeros(len(classes))
    for split in SPLITS:
        values = np.array([counts[split][index] for index in range(len(classes))])
        axis.bar(range(len(classes)), values, bottom=bottom, label=f'{split} ({values.sum():,})', width=0.85)
        bottom += values
    axis.set_xticks(range(len(classes)), [name.replace('_', ' ') for name in classes], rotation=90, fontsize=5)
    axis.set_xlim(-1, len(classes))
    axis.set_ylabel('Images')
    axis.set_title('Images per class')
    axis.legend(fontsize=8, loc='lower right')
    figure.tight_layout()
    figure.savefig(out / 'class_counts.png', dpi=300)
    plt.close(figure)

    return {
        split: {
            'images': sum(counts[split].values()),
            'per_class_min': min(counts[split].values()),
            'per_class_max': max(counts[split].values()),
        }
        for split in SPLITS
    }


def _threaded(function, items, workers, label):
    '''map() over threads, in order, printing progress every 5,000 items.

    Threads rather than processes: the work is mostly waiting on the volume
    and on JPEG decoding, which release the GIL, and forking after torch is
    imported can hang.
    '''
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, result in enumerate(pool.map(function, items), start=1):
            results.append(result)
            if done % 5000 == 0 or done == len(items):
                print(f'{label}: {done}/{len(items)}', flush=True)
    return results


def _sample(count, sample_size, seed):
    '''Sorted indices of a fixed random sample; every index when sample_size is 0.'''
    if not 0 < sample_size < count:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, size=sample_size, replace=False))


def _header(path):
    # Image.open reads only the header; the pixels are not decoded.
    with Image.open(path) as image:
        return image.width, image.height, image.mode


def image_sizes(raw_root, workers, out, sample_size=0, seed=0):
    '''Size statistics of the original images, all of them unless sample_size > 0.'''
    paths = []
    for split in SPLITS:
        dataset = Food101(raw_root, split=split)
        paths += [dataset.images_folder / f'{image_id}.jpg' for image_id in dataset.ids]

    population = len(paths)
    paths = [paths[index] for index in _sample(population, sample_size, seed)]
    headers = _threaded(_header, paths, workers, 'image headers')

    width = np.array([w for w, _, _ in headers])
    height = np.array([h for _, h, _ in headers])
    shorter = np.minimum(width, height)
    longer = np.maximum(width, height)

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    axes[0].hexbin(width, height, gridsize=40, mincnt=1, bins='log', cmap='viridis')
    axes[0].set_xlabel('Width (px)')
    axes[0].set_ylabel('Height (px)')
    axes[0].set_title('Original image sizes')
    axes[1].hist(shorter, bins=50)
    axes[1].axvline(256, color='C3', linestyle='--', label='256 px (before crop)')
    axes[1].set_xlabel('Shorter side (px)')
    axes[1].set_title('Shorter side')
    axes[1].legend(fontsize=8)
    axes[2].hist(width / height, bins=50)
    axes[2].set_xlabel('Width / height')
    axes[2].set_title('Aspect ratio')
    figure.tight_layout()
    figure.savefig(out / 'image_sizes.png', dpi=300)
    plt.close(figure)

    return {
        'images': len(headers),
        'population': population,
        'modes': dict(Counter(mode for _, _, mode in headers)),
        'longer_side': {'min': int(longer.min()), 'median': float(np.median(longer)), 'max': int(longer.max())},
        'shorter_side': {'min': int(shorter.min()), 'median': float(np.median(shorter)), 'max': int(shorter.max())},
        'shorter_side_below_256': int((shorter < 256).sum()),
        'aspect_ratio': {
            'min': float((width / height).min()),
            'median': float(np.median(width / height)),
            'max': float((width / height).max()),
        },
    }


def samples(dataset, out, classes_shown=8, per_class=6, seed=0):
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    shown = sorted(rng.choice(np.unique(targets), size=min(classes_shown, len(np.unique(targets))), replace=False))
    crop = pipeline(EVAL_STEPS)

    figure, axes = plt.subplots(len(shown), per_class, figsize=(1.6 * per_class, 1.75 * len(shown)), squeeze=False)
    for row, label in enumerate(shown):
        members = np.flatnonzero(targets == label)
        picks = rng.choice(members, size=min(per_class, len(members)), replace=False)
        for col in range(per_class):
            axis = axes[row][col]
            axis.set_axis_off()
            if col < len(picks):
                axis.imshow(crop(dataset.load_image(picks[col])))
        axes[row][0].set_title(dataset.classes[label].replace('_', ' '), fontsize=8, loc='left')
    figure.tight_layout()
    # PNG keeps the rendered photo grid lossless.
    figure.savefig(out / 'samples.png', dpi=200)
    plt.close(figure)


def augmentation(dataset, out, images=4, versions=6, seed=0):
    '''Each row: the stored center crop, then the same image through the train pipeline.'''
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    picks = rng.choice(len(dataset), size=min(images, len(dataset)), replace=False)
    crop, augment = pipeline(EVAL_STEPS), pipeline(TRAIN_STEPS)

    figure, axes = plt.subplots(len(picks), versions + 1, figsize=(1.6 * (versions + 1), 1.7 * len(picks)),
                                squeeze=False)
    for row, index in enumerate(picks):
        image = dataset.load_image(index)
        axes[row][0].imshow(crop(image))
        for col in range(1, versions + 1):
            axes[row][col].imshow(augment(image))
        for axis in axes[row]:
            axis.set_axis_off()
    axes[0][0].set_title('fixed crop', fontsize=8)
    for col in range(1, versions + 1):
        axes[0][col].set_title(f'train #{col}', fontsize=8)
    figure.suptitle('flip + rotate 15° + ColorJitter + translate 10%', fontsize=9)
    figure.tight_layout()
    figure.savefig(out / 'augmentation.png', dpi=200)
    plt.close(figure)


def channel_stats(dataset, workers, sample_size=0, seed=0):
    '''Per-channel mean and std of the stored train crops, all of them unless sample_size > 0.'''
    crop = pipeline(EVAL_STEPS)

    def sums(index):
        x = np.asarray(crop(dataset.load_image(index)), dtype=np.float64) / 255.0
        return x.sum(axis=(0, 1)), (x ** 2).sum(axis=(0, 1)), x.shape[0] * x.shape[1]

    picks = _sample(len(dataset), sample_size, seed)
    parts = _threaded(sums, list(picks), workers, 'channel stats')
    pixels = sum(count for _, _, count in parts)
    mean = sum(total for total, _, _ in parts) / pixels
    std = np.sqrt(sum(total_sq for _, total_sq, _ in parts) / pixels - mean ** 2)
    return {'sample_images': len(picks), 'mean': mean.tolist(), 'std': std.tolist()}


def write_summary(summary, results):
    (results / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    counts = summary['splits']
    lines = [
        '| Split | Images | Per class |',
        '|---|---|---|',
    ]
    for split in SPLITS:
        info = counts[split]
        per_class = str(info['per_class_min']) if info['per_class_min'] == info['per_class_max'] \
            else f"{info['per_class_min']}–{info['per_class_max']}"
        lines.append(f"| {split} | {info['images']:,} | {per_class} |")

    if 'sizes' in summary:
        sizes = summary['sizes']
        lines += [
            '',
            f"Original images measured: {sizes['images']:,} of {sizes['population']:,}; modes {sizes['modes']}.",
            f"Longer side {sizes['longer_side']['min']}–{sizes['longer_side']['max']} px "
            f"(median {sizes['longer_side']['median']:.0f}); shorter side {sizes['shorter_side']['min']}–"
            f"{sizes['shorter_side']['max']} px (median {sizes['shorter_side']['median']:.0f}); "
            f"{sizes['shorter_side_below_256']:,} images have a shorter side below 256 px.",
        ]

    stats = summary['channels']
    fmt = ', '.join
    lines += [
        '',
        f"Channel mean of {stats['sample_images']:,} train crops: [{fmt(f'{v:.3f}' for v in stats['mean'])}], "
        f"std: [{fmt(f'{v:.3f}' for v in stats['std'])}].",
        f'Pipeline normalizes with mean {MEAN}, std {STD}; ImageNet weights expect mean {IMAGENET_MEAN}, '
        f'std {IMAGENET_STD}.',
    ]
    (results / 'summary.md').write_text('\n'.join(lines) + '\n')


def _n(value):
    '''An integer with thin-space thousands, as LaTeX prints 68\\,175.'''
    return f'{value:,}'.replace(',', '\\,')


def write_tex(summary, results):
    '''Booktabs tables that reports/assignment_1.tex pulls in with \\input.'''
    rows = []
    for split in SPLITS:
        info = summary['splits'][split]
        per_class = str(info['per_class_min']) if info['per_class_min'] == info['per_class_max'] \
            else f"{info['per_class_min']}--{info['per_class_max']}"
        rows.append(f'{split} & {_n(info["images"])} & {per_class} \\\\')
    total = sum(summary['splits'][split]['images'] for split in SPLITS)
    (results / 'splits.tex').write_text('\n'.join([
        r'\begin{tabular}{lrr}',
        r'\toprule',
        r'Tập & Số ảnh & Ảnh/lớp \\',
        r'\midrule',
        *rows,
        r'\midrule',
        f'tổng & {_n(total)} & \\\\',
        r'\bottomrule',
        r'\end{tabular}',
        '',
    ]))

    stats = summary['channels']
    triple = ', '.join
    lines = [r'\begin{tabular}{ll}', r'\toprule', r'Đại lượng & Giá trị \\', r'\midrule']
    if 'sizes' in summary:
        sizes = summary['sizes']
        modes = ', '.join(f'{mode} {_n(count)}' for mode, count in sorted(sizes['modes'].items()))
        size_label = 'Ảnh gốc' if sizes['images'] == sizes['population'] else \
            f'Ảnh gốc (mẫu {_n(sizes["images"])}/{_n(sizes["population"])})'
        lines += [
            f'{size_label} & {_n(sizes["images"])} ({modes}) \\\\',
            f'Cạnh dài (px) & {sizes["longer_side"]["min"]}--{sizes["longer_side"]["max"]}, '
            f'trung vị {sizes["longer_side"]["median"]:.0f} \\\\',
            f'Cạnh ngắn (px) & {sizes["shorter_side"]["min"]}--{sizes["shorter_side"]["max"]}, '
            f'trung vị {sizes["shorter_side"]["median"]:.0f} \\\\',
            f'Cạnh ngắn \\(<256\\) px & {_n(sizes["shorter_side_below_256"])} ảnh \\\\',
            f'Tỉ lệ rộng/cao & {sizes["aspect_ratio"]["min"]:.2f}--{sizes["aspect_ratio"]["max"]:.2f}, '
            f'trung vị {sizes["aspect_ratio"]["median"]:.2f} \\\\',
        ]
    lines += [
        f'Mean RGB ({_n(stats["sample_images"])} ảnh train) & '
        f'({triple(f"{v:.3f}" for v in stats["mean"])}) \\\\',
        f'Std RGB & ({triple(f"{v:.3f}" for v in stats["std"])}) \\\\',
        r'\bottomrule',
        r'\end{tabular}',
        '',
    ]
    (results / 'stats.tex').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Food-101 EDA for A1.')
    parser.add_argument('--root', default='data/food101_256', help='prepared copy the models train on')
    parser.add_argument('--raw', default='data/', help='original download; skip size statistics if missing')
    parser.add_argument('--out', default=str(_repo_root / 'docs' / 'assets' / 'a1' / 'eda'),
                        help='figure directory')
    parser.add_argument('--results-dir', default=str(_repo_root / 'reports' / 'results' / 'a1' / 'eda'),
                        help='CSV and LaTeX table directory')
    parser.add_argument('--workers', type=int, default=32, help='threads reading images')
    parser.add_argument('--stats-sample', type=int, default=0, help='train images for the mean/std; 0 for all')
    parser.add_argument('--size-sample', type=int, default=0, help='original images measured; 0 for all')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.root)
    summary = {'splits': class_counts(splits, out, results)}
    print('class counts done', flush=True)

    if (Path(args.raw) / 'food-101' / 'images').is_dir():
        summary['sizes'] = image_sizes(args.raw, args.workers, out, args.size_sample)
        print('image sizes done', flush=True)
    else:
        print(f'skip image sizes: no original download under {args.raw}')

    samples(splits['train'], out)
    print('samples done', flush=True)
    augmentation(splits['train'], out)
    print('augmentation done', flush=True)
    summary['channels'] = channel_stats(splits['train'], args.workers, args.stats_sample)
    write_summary(summary, results)
    write_tex(summary, results)
    print((results / 'summary.md').read_text())
    print(f'wrote EDA figures to {out}')
    print(f'wrote EDA tables to {results}')


if __name__ == '__main__':
    main()
