import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from datasets.mnist import MNIST

N_CLASSES = 10
SPLITS = (('train', 'Train'), ('val', 'Validation'), ('test', 'Test'))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else _repo_root / path


def _targets(dataset):
    indices = np.arange(len(dataset)) if dataset.indices is None else np.asarray(dataset.indices)
    return dataset.targets.numpy()[indices]


def sample_grid(train, per_class, seed, path):
    targets = _targets(train)
    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(
        per_class, N_CLASSES,
        figsize=(1.2 * N_CLASSES, 1.2 * per_class),
        squeeze=False,
    )

    for label in range(N_CLASSES):
        members = np.flatnonzero(targets == label)
        picks = rng.choice(members, size=min(per_class, len(members)), replace=False)

        for row in range(per_class):
            axis = axes[row][label]
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_frame_on(False)

            if row < len(picks):
                image, _ = train[int(picks[row])]
                axis.imshow(image, cmap='gray', vmin=0, vmax=255)

    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(figure)


def augmentation_grid(train, seed, path):
    targets = _targets(train)
    rng = np.random.default_rng(seed)
    originals = [
        train[int(rng.choice(np.flatnonzero(targets == label)))][0]
        for label in range(N_CLASSES)
    ]
    transform = transforms.RandomAffine(10, translate=(0.1, 0.1))
    torch.manual_seed(seed)
    augmented = [transform(image) for image in originals]
    figure, axes = plt.subplots(2, N_CLASSES, figsize=(12, 3.2), squeeze=False)

    for label, (original, changed) in enumerate(zip(originals, augmented)):
        for row, image in enumerate((original, changed)):
            axis = axes[row][label]
            axis.imshow(image, cmap='gray', vmin=0, vmax=255)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_frame_on(False)

    axes[0][0].set_ylabel('Ảnh gốc', rotation=0, labelpad=30, fontsize=10, va='center')
    axes[1][0].set_ylabel('Tăng cường', rotation=0, labelpad=30, fontsize=10, va='center')
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(figure)


def class_counts(splits):
    return {key: np.bincount(_targets(dataset), minlength=N_CLASSES) for key, dataset in splits.items()}


def write_split_table(counts, results_dir):
    lines = [f"{'Class':<6}{'Train':>8}{'Validation':>12}{'Test':>8}"]

    for digit in range(N_CLASSES):
        values = [int(counts[key][digit]) for key, _ in SPLITS]
        lines.append(f'{digit:<6}{values[0]:>8,}{values[1]:>12,}{values[2]:>8,}')

    values = [int(counts[key].sum()) for key, _ in SPLITS]
    lines.append(f'{"Total":<6}{values[0]:>8,}{values[1]:>12,}{values[2]:>8,}')

    path = results_dir / 'split_counts.txt'
    path.write_text('\n'.join(lines) + '\n')
    return {key: int(counts[key].sum()) for key, _ in SPLITS}


def plot_class_balance(counts, path):
    label_mapping = {
        'Train': 'Huấn luyện',
        'Validation': 'Kiểm định',
        'Test': 'Kiểm thử',
    }
    digits = np.arange(N_CLASSES)
    width = 0.26
    figure, axis = plt.subplots(figsize=(12, 4.2))

    for index, (key, label) in enumerate(SPLITS):
        values = counts[key] / counts[key].sum() * 100
        offset = (index - 1) * width
        axis.bar(
            digits + offset, values, width,
            label=f'{label_mapping[label]}',
        )

    axis.axhline(10, color='black', linestyle='--', linewidth=1, label='cân bằng (10%)')
    axis.set_xticks(digits)
    axis.set_xlabel('Lớp')
    axis.set_ylabel('Tỉ lệ (%)')
    axis.set_ylim(0, 13)
    axis.grid(axis='y', alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.16), frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Generate MNIST EDA figures and tables.')
    parser.add_argument('--root', default='data/', help='MNIST root directory')
    parser.add_argument('--per-class', type=int, default=8, help='samples per digit')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--figures-dir', default=str(_repo_root / 'docs' / 'assets' / 'shared'),
                        help='figure output directory')
    parser.add_argument('--results-dir', default=str(_repo_root / 'reports' / 'results' / 'shared'),
                        help='plain-text table directory')
    args = parser.parse_args()

    if args.per_class < 1:
        raise SystemExit('--per-class must be positive')

    figures_dir = _resolve(args.figures_dir)
    results_dir = _resolve(args.results_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    root = str(_resolve(args.root))
    splits = {key: MNIST(root=root, split=key) for key, _ in SPLITS}
    counts = class_counts(splits)

    path = figures_dir / 'mnist_samples.png'
    sample_grid(splits['train'], args.per_class, args.seed, path)
    print(f'wrote {path}')

    path = figures_dir / 'mnist_augmentation.png'
    augmentation_grid(splits['train'], args.seed, path)
    print(f'wrote {path}')

    path = figures_dir / 'class_balance.png'
    plot_class_balance(counts, path)
    print(f'wrote {path}')

    totals = write_split_table(counts, results_dir)
    print('samples: ' + ', '.join(f'{label}={totals[key]:,}' for key, label in SPLITS))
    print(f'wrote {results_dir / "split_counts.txt"}')


if __name__ == '__main__':
    main()
