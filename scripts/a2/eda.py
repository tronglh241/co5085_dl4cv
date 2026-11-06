import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from datasets.celeba import CelebA
from scripts.a2.result_io import write_result

SPLITS = ('train', 'val', 'test')


def load_splits(root):
    return {split: CelebA(root, split=split) for split in SPLITS}


def write_counts(splits, results, out):
    path = results / 'split_counts.csv'
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['split', 'images', 'identities'])
        for split, dataset in splits.items():
            writer.writerow([split, len(dataset), len(set(dataset.identities))])

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    labels = list(splits)
    axes[0].bar(labels, [len(dataset) for dataset in splits.values()])
    axes[0].set_title('Images')
    axes[1].bar(labels, [len(set(dataset.identities)) for dataset in splits.values()])
    axes[1].set_title('Identities')
    for axis in axes:
        axis.set_yscale('log')
    figure.tight_layout()
    figure.savefig(out / 'split_counts.png', dpi=300)
    plt.close(figure)


def plot_identity_hist(train, out):
    counts = np.array(list(Counter(train.identities).values()))
    figure, axis = plt.subplots(figsize=(6, 3.4))
    axis.hist(counts, bins=50)
    axis.set_yscale('log')
    axis.set_xlabel('Images per identity')
    axis.set_ylabel('Identities')
    axis.set_title('Train identities')
    figure.tight_layout()
    figure.savefig(out / 'identity_hist.png', dpi=300)
    plt.close(figure)
    return {
        'min': int(counts.min()),
        'median': float(np.median(counts)),
        'max': int(counts.max()),
        'single_image': int((counts == 1).sum()),
    }


def plot_samples(train, out, seed, identities=6, per_identity=4):
    groups = defaultdict(list)
    for index, identity in enumerate(train.identities):
        groups[identity].append(index)

    eligible = sorted(identity for identity, members in groups.items() if len(members) >= per_identity)
    rng = np.random.default_rng(seed)
    shown = rng.choice(eligible, size=min(identities, len(eligible)), replace=False)

    figure, axes = plt.subplots(
        len(shown),
        per_identity,
        figsize=(1.6 * per_identity, 1.75 * len(shown)),
        squeeze=False,
    )
    for row, identity in enumerate(shown):
        members = groups[identity]
        picks = rng.choice(members, size=per_identity, replace=False)
        for col, index in enumerate(picks):
            axes[row][col].imshow(train.load_image(index))
            axes[row][col].set_axis_off()
        axes[row][0].set_title(f'identity {identity}', fontsize=8, loc='left')
    figure.tight_layout()
    figure.savefig(out / 'samples.png', dpi=200)
    plt.close(figure)


def plot_alignment(train, raw_root, out, seed, count=4):
    source_folder = raw_root / 'celeba' / 'img_align_celeba'
    if not source_folder.is_dir():
        print(f'skip alignment preview: {source_folder} is missing')
        return False

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(train), size=min(count, len(train)), replace=False)
    figure, axes = plt.subplots(len(picks), 2, figsize=(4.2, 2.1 * len(picks)), squeeze=False)

    for row, index in enumerate(picks):
        image_id = train.ids[index]
        with Image.open(source_folder / image_id) as image:
            axes[row][0].imshow(image.convert('RGB'), aspect='equal')
        axes[row][0].set_title('raw img_align_celeba', fontsize=8)
        axes[row][1].imshow(train.load_image(index), aspect='equal')
        axes[row][1].set_title('aligned 112x112', fontsize=8)
        for axis in axes[row]:
            axis.set_axis_off()

    figure.tight_layout()
    figure.savefig(out / 'alignment.png', dpi=200)
    plt.close(figure)
    return True


def main():
    parser = argparse.ArgumentParser(description='EDA for the A2 CelebA copy.')
    parser.add_argument('--root', default='data/celeba_112')
    parser.add_argument('--raw', default='data/celeba')
    parser.add_argument('--out', default='docs/assets/a2/eda')
    parser.add_argument('--results-dir', default='reports/results/a2/eda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    results = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    splits = load_splits(args.root)
    write_counts(splits, results, out)
    identity = plot_identity_hist(splits['train'], out)
    plot_samples(splits['train'], out, args.seed)
    aligned = plot_alignment(splits['train'], Path(args.raw), out, args.seed)

    summary = {
        'splits': {
            split: {
                'images': len(dataset),
                'identities': len(set(dataset.identities)),
            }
            for split, dataset in splits.items()
        },
        'train_images_per_identity': identity,
        'alignment_preview': aligned,
    }
    write_result(results / 'summary.txt', summary)
    print(f'wrote EDA outputs to {out} and {results}')


if __name__ == '__main__':
    main()
