import csv
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import MNIST as TorchvisionMNIST

from datasets.mnist import MNIST

ASSETS = _repo_root / 'docs' / 'assets' / 'shared'
RESULTS = _repo_root / 'reports' / 'results' / 'shared'
N_CLASSES = 10


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def load_splits():
    train = MNIST(root='data/', split='train')
    val = MNIST(root='data/', split='val')
    test = MNIST(root='data/', split='test')
    return train, val, test


def original():
    train_raw = TorchvisionMNIST(root='data/', train=True, download=True)
    test_raw = TorchvisionMNIST(root='data/', train=False, download=True)

    print('\n== Original ==')
    print(f'Number of samples of train: {len(train_raw)}')
    print(f'Number of samples of test: {len(test_raw)}')

    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(['train', 'test'], [len(train_raw), len(test_raw)])
    axis.bar_label(bars, labels=[f'{len(train_raw):,}', f'{len(test_raw):,}'])
    axis.set_ylabel('images')
    figure.tight_layout()
    figure.savefig(ASSETS / 'original_counts.png', bbox_inches='tight', dpi=300)
    plt.close(figure)

    return {'original_train': len(train_raw), 'original_test': len(test_raw)}


def overview(train, val, test):
    total = len(train) + len(val) + len(test)
    raw = _repo_root / 'data' / 'MNIST' / 'raw'
    size_mb = sum(path.stat().st_size for path in raw.iterdir()) / 1e6
    print('\n== Overview ==')
    print(f'Total number of samples: {total} ({N_CLASSES} digit classes 0-9, {size_mb:.1f}MB raw)')
    return {'total': total, 'raw_size_mb': size_mb}


def split(train, val, test):
    total = len(train) + len(val) + len(test)
    print('\n== Split (stratified) ==')
    print(f'Number of samples of train: {len(train)} ({len(train) / total:.1%})')
    print(f'Number of samples of val: {len(val)} ({len(val) / total:.1%})')
    print(f'Number of samples of test: {len(test)} ({len(test) / total:.1%})')
    return {
        'train': len(train),
        'val': len(val),
        'test': len(test),
        'train_share': len(train) / total,
        'val_share': len(val) / total,
        'test_share': len(test) / total,
    }


def split_plot(train, val, test):
    names = ['train', 'val', 'test']
    sizes = [len(train), len(val), len(test)]
    total = sum(sizes)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(names, sizes)
    axis.bar_label(bars, labels=[f'{count:,}' for count in sizes])
    axis.set_ylabel('images')
    figure.savefig(ASSETS / 'split_counts.png', dpi=300)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    wedges, _, _ = axis.pie([size / total for size in sizes], autopct='%1.1f%%', startangle=90)
    axis.legend(wedges, names, loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=3)
    figure.tight_layout()
    figure.savefig(ASSETS / 'split_share.png', bbox_inches='tight', dpi=300)
    plt.close(figure)


def geometry(train, val, test):
    print('\n== Geometry ==')
    info = {}
    for name, dataset in (('train', train), ('val', val), ('test', test)):
        sizes, modes, lo, hi = set(), set(), 255, 0
        for index in range(len(dataset)):
            image, _ = dataset[index]
            sizes.add(image.size)
            modes.add(image.mode)
            lo = min(lo, image.getextrema()[0])
            hi = max(hi, image.getextrema()[1])
        sizes_str = ', '.join(f'{width}x{height}' for width, height in sorted(sizes))
        modes_str = ', '.join(sorted(modes))
        print(f'Number of samples of {name}: {len(dataset)}')
        print(f'  image sizes: {sizes_str}; mode: {modes_str}; pixel range: [{lo}, {hi}]')
        info = {'image_sizes': sizes_str, 'image_mode': modes_str, 'pixel_min': lo, 'pixel_max': hi}
    return info


def normalization(train):
    pixels = train.data[torch.tensor(train.indices)].float() / 255.0
    mean = pixels.mean().item()
    std = pixels.std().item()
    print('\n== Normalization ==')
    print(f'Measured over train: mean={mean:.4f} std={std:.4f}')
    return {'mean': mean, 'std': std}


def sample_grid(train, per_digit=8):
    print('\n== Samples ==')
    by_digit = {digit: [] for digit in range(N_CLASSES)}

    for index in range(len(train)):
        digit = train[index][1]
        if isinstance(digit, torch.Tensor):
            digit = digit.item()
        if len(by_digit[digit]) < per_digit:
            by_digit[digit].append(index)
        if all(len(indices) == per_digit for indices in by_digit.values()):
            break

    figure, axes = plt.subplots(10, per_digit, figsize=(8, 10))
    for digit in range(N_CLASSES):
        for column, index in enumerate(by_digit[digit]):
            axes[digit, column].imshow(train[index][0], cmap='gray')
            axes[digit, column].set_xticks([])
            axes[digit, column].set_yticks([])
        axes[digit, 0].set_ylabel(str(digit), rotation=0, labelpad=12, fontsize=12)

    figure.tight_layout()
    figure.savefig(ASSETS / 'samples.png', dpi=300)
    plt.close(figure)


def augmentation(train):
    indices = list(range(5)) if train.indices is None else list(train.indices[:5])
    images = [Image.fromarray(train.data[index].numpy()) for index in indices]
    transform = transforms.RandomAffine(10, translate=(0.1, 0.1))

    figure, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for axis, image in zip(axes[0], images):
        axis.imshow(image, cmap='gray')
        axis.set_axis_off()
    axes[0, 0].set_title('original', fontsize=9)
    for axis, image in zip(axes[1], images):
        axis.imshow(transform(image), cmap='gray')
        axis.set_axis_off()
    axes[1, 0].set_title('RandomAffine(10, translate=0.1)', fontsize=9)
    figure.tight_layout()
    figure.savefig(ASSETS / 'augmentation.png', dpi=300)
    plt.close(figure)


def class_balance(train, val, test):
    print('\n== Class Balance ==')
    counts = {}

    for name, dataset in (('train', train), ('val', val), ('test', test)):
        per_digit = [0] * N_CLASSES
        for index in range(len(dataset)):
            per_digit[dataset[index][1]] += 1
        counts[name] = per_digit

    ratio = max(counts['train']) / min(counts['train'])
    for name in ('train', 'val', 'test'):
        print(f'Number of samples per digit of {name}: {counts[name]}')
    print(f'Imbalance ratio (train max/min): {ratio:.2f}x (balanced under 2.0x)')

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for offset, name in enumerate(('train', 'val', 'test')):
        axis.bar(
            [digit + (offset - 1) * 0.25 for digit in range(N_CLASSES)],
            counts[name],
            width=0.25,
            label=name,
        )
    axis.set_xlabel('digit')
    axis.set_ylabel('images')
    axis.set_xticks(list(range(N_CLASSES)))
    axis.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=3)
    figure.tight_layout()
    figure.savefig(ASSETS / 'class_balance.png', bbox_inches='tight', dpi=300)
    plt.close(figure)

    return {'counts': counts, 'ratio': ratio}


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    train, val, test = load_splits()

    original_counts = original()
    overview_counts = overview(train, val, test)
    split_counts = split(train, val, test)
    split_plot(train, val, test)
    geometry_info = geometry(train, val, test)
    normalization_info = normalization(train)
    sample_grid(train)
    augmentation(train)
    balance = class_balance(train, val, test)

    rows = [
        {'Metric': 'original train samples', 'Value': original_counts['original_train']},
        {'Metric': 'original test samples', 'Value': original_counts['original_test']},
        {'Metric': 'total samples', 'Value': overview_counts['total']},
        {'Metric': 'raw size (MB)', 'Value': f"{overview_counts['raw_size_mb']:.1f}"},
        {'Metric': 'train samples', 'Value': split_counts['train']},
        {'Metric': 'val samples', 'Value': split_counts['val']},
        {'Metric': 'test samples', 'Value': split_counts['test']},
        {'Metric': 'train share', 'Value': f"{split_counts['train_share']:.1%}"},
        {'Metric': 'val share', 'Value': f"{split_counts['val_share']:.1%}"},
        {'Metric': 'test share', 'Value': f"{split_counts['test_share']:.1%}"},
        {'Metric': 'image sizes', 'Value': geometry_info['image_sizes']},
        {'Metric': 'image mode', 'Value': geometry_info['image_mode']},
        {'Metric': 'pixel range', 'Value': f"[{geometry_info['pixel_min']}, {geometry_info['pixel_max']}]"},
        {'Metric': 'train mean', 'Value': f"{normalization_info['mean']:.4f}"},
        {'Metric': 'train std', 'Value': f"{normalization_info['std']:.4f}"},
        {'Metric': 'train per-digit min', 'Value': min(balance['counts']['train'])},
        {'Metric': 'train per-digit max', 'Value': max(balance['counts']['train'])},
        {'Metric': 'imbalance ratio', 'Value': f"{balance['ratio']:.2f}"},
    ]
    write_csv(RESULTS / 'eda_summary.csv', ('Metric', 'Value'), rows)
    print(f'\nwrote {RESULTS / "eda_summary.csv"}')


if __name__ == '__main__':
    main()
