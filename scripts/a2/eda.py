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
import torch
from PIL import Image
from torchvision import transforms

from datasets.celeba import CelebA
from scripts.a2.result_io import write_result

SPLITS = ('train', 'val', 'test')

# Keep this in sync with the train transforms in configs/a2/train_*.yml.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_splits(root):
    return {split: CelebA(root, split=split) for split in SPLITS}


def identity_stats(dataset):
    counts = np.array(list(Counter(dataset.identities).values()))
    return {
        'images': len(dataset),
        'identities': len(set(dataset.identities)),
        'min': int(counts.min()),
        'median': float(np.median(counts)),
        'mean': float(counts.mean()),
        'max': int(counts.max()),
        'single_image': int((counts == 1).sum()),
    }


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
    median = float(np.median(counts))
    mean = float(counts.mean())
    single_image = int((counts == 1).sum())

    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    bins = np.arange(0.5, counts.max() + 1.5, 1)
    axis.hist(
        counts,
        bins=bins,
        color='tab:blue',
        alpha=0.8,
    )
    axis.axvline(
        median,
        color='tab:orange',
        linestyle='--',
        linewidth=1.6,
        label=f'Trung vị = {median:.0f}',
    )
    axis.axvline(
        mean,
        color='tab:green',
        linestyle=':',
        linewidth=1.6,
        label=f'Trung bình = {mean:.1f}',
    )
    axis.set_yscale('linear')
    axis.set_xlabel('Số ảnh trên mỗi danh tính')
    axis.set_ylabel('Số danh tính')
    axis.set_title('Phân bố số ảnh trên mỗi danh tính ở tập train')
    axis.grid(axis='y', alpha=0.25)
    axis.set_xlim(0, counts.max() + 1)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(out / 'identity_hist.png', dpi=300)
    plt.close(figure)
    return identity_stats(train)


def plot_samples(train, raw_root, out, seed, identities=6, per_identity=4):
    source_folder = Path(raw_root) / 'celeba' / 'img_align_celeba'
    if not source_folder.is_dir():
        print(f'skip sample preview: {source_folder} is missing')
        return False

    groups = defaultdict(list)
    for index, identity in enumerate(train.identities):
        groups[identity].append(index)

    eligible = sorted(identity for identity, members in groups.items() if len(members) >= per_identity)
    rng = np.random.default_rng(seed)
    shown = rng.choice(eligible, size=min(identities, len(eligible)), replace=False)

    figure, axes = plt.subplots(
        len(shown),
        per_identity,
        figsize=(1.2 * per_identity, 1.47 * len(shown)),
        squeeze=False,
    )
    for row, identity in enumerate(shown):
        members = groups[identity]
        picks = rng.choice(members, size=per_identity, replace=False)
        for col, index in enumerate(picks):
            image_id = train.ids[index]
            with Image.open(source_folder / image_id) as image:
                axes[row][col].imshow(image.convert('RGB'))
            axes[row][col].set_axis_off()
        axes[row][0].set_title(f'Danh tính {identity}', fontsize=8, loc='left')
    figure.tight_layout(pad=0.1, w_pad=0.02, h_pad=0.25)
    figure.savefig(out / 'samples.png', dpi=200)
    plt.close(figure)
    return True


def plot_alignment(train, raw_root, out, seed, count=4):
    source_folder = raw_root / 'celeba' / 'img_align_celeba'
    if not source_folder.is_dir():
        print(f'skip alignment preview: {source_folder} is missing')
        return False

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(train), size=min(count, len(train)), replace=False)
    figure, axes = plt.subplots(2, len(picks), figsize=(2.1 * len(picks), 4.2), squeeze=False)

    for col, index in enumerate(picks):
        image_id = train.ids[index]
        with Image.open(source_folder / image_id) as image:
            axes[0][col].imshow(image.convert('RGB'), aspect='equal')
        axes[1][col].imshow(train.load_image(index), aspect='equal')
        for row in axes:
            row[col].set_axis_off()

    axes[0][0].set_title('Trước căn chỉnh', fontsize=9, loc='left')
    axes[1][0].set_title('Sau căn chỉnh', fontsize=9, loc='left')
    figure.tight_layout(pad=0.2, w_pad=0.05, h_pad=0.35)
    figure.savefig(out / 'alignment.png', dpi=200)
    plt.close(figure)
    return True


def plot_augmentation(train, out, seed, count=4):
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(train), size=min(count, len(train)), replace=False)

    figure, axes = plt.subplots(2, len(picks), figsize=(2.1 * len(picks), 4.2), squeeze=False)
    for col, index in enumerate(picks):
        image = train.load_image(index)

        torch.manual_seed(seed + int(index))
        augmented = TRAIN_TRANSFORM(image)
        augmented = augmented * 0.5 + 0.5
        augmented = augmented.clamp(0, 1).permute(1, 2, 0).numpy()

        axes[0][col].imshow(image)
        axes[1][col].imshow(augmented)
        for row in axes:
            row[col].set_axis_off()

    axes[0][0].set_title('Trước augmentation', fontsize=9, loc='left')
    axes[1][0].set_title('Sau augmentation', fontsize=9, loc='left')
    figure.tight_layout(pad=0.2, w_pad=0.05, h_pad=0.35)
    figure.savefig(out / 'augmentation.png', dpi=200)
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
    val_identity = identity_stats(splits['val'])
    test_identity = identity_stats(splits['test'])
    plot_samples(splits['train'], Path(args.raw), out, args.seed)
    aligned = plot_alignment(splits['train'], Path(args.raw), out, args.seed)
    augmented = plot_augmentation(splits['train'], out, args.seed)

    summary = {
        'splits': {
            split: {
                'images': len(dataset),
                'identities': len(set(dataset.identities)),
            }
            for split, dataset in splits.items()
        },
        'train_images_per_identity': identity,
        'val_images_per_identity': val_identity,
        'test_images_per_identity': test_identity,
        'alignment_preview': aligned,
        'augmentation_preview': augmented,
    }
    write_result(results / 'summary.txt', summary)
    print(f'wrote EDA outputs to {out} and {results}')


if __name__ == '__main__':
    main()
