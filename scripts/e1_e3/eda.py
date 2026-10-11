import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST as TorchvisionMNIST

from datasets.mnist import MNIST

ASSETS = Path('reports', 'images', 'e1_e3')
N_CLASSES = 10


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

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(['train', 'test'], [len(train_raw), len(test_raw)])

    ax.bar_label(bars, labels=[f'{len(train_raw):,}', f'{len(test_raw):,}'])
    ax.set_ylabel('images')
    fig.tight_layout()
    fig.savefig(ASSETS / 'original_counts.pdf', bbox_inches='tight')
    plt.close(fig)


def overview(train, val, test):
    total = len(train) + len(val) + len(test)
    raw = Path('data', 'MNIST', 'raw')
    size_mb = sum(f.stat().st_size for f in raw.iterdir()) / 1e6
    print('\n== Overview ==')
    print(f'Total number of samples: {total} ({N_CLASSES} digit classes 0-9, {size_mb:.1f}MB raw)')


def split(train, val, test):
    total = len(train) + len(val) + len(test)
    print('\n== Split (stratified) ==')
    print(f'Number of samples of train: {len(train)} ({len(train) / total:.1%})')
    print(f'Number of samples of val: {len(val)} ({len(val) / total:.1%})')
    print(f'Number of samples of test: {len(test)} ({len(test) / total:.1%})')


def split_plot(train, val, test):
    names = ['train', 'val', 'test']
    sizes = [len(train), len(val), len(test)]
    total = sum(sizes)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(names, sizes)

    ax.bar_label(bars, labels=[f'{c:,}' for c in sizes])
    ax.set_ylabel('images')
    fig.savefig(ASSETS / 'split_counts.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, _, _ = ax.pie([s / total for s in sizes], autopct='%1.1f%%', startangle=90)

    ax.legend(wedges, names, loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=3)
    fig.tight_layout()
    fig.savefig(ASSETS / 'split_share.pdf', bbox_inches='tight')
    plt.close(fig)


def geometry(train, val, test):
    print('\n== Geometry ==')
    for name, dataset in (('train', train), ('val', val), ('test', test)):
        sizes, modes, lo, hi = set(), set(), 255, 0
        for i in range(len(dataset)):
            img, _ = dataset[i]
            sizes.add(img.size)
            modes.add(img.mode)
            lo = min(lo, img.getextrema()[0])
            hi = max(hi, img.getextrema()[1])
        sizes_str = ', '.join(f'{w}x{h}' for w, h in sorted(sizes))
        modes_str = ', '.join(sorted(modes))
        print(f'Number of samples of {name}: {len(dataset)}')
        print(f'  image sizes: {sizes_str}; mode: {modes_str}; pixel range: [{lo}, {hi}]')


def normalization(train):
    print('\n== Normalization ==')
    px = train.data[torch.tensor(train.indices)].float() / 255.0
    print(f'Measured over train: mean={px.mean().item():.4f} std={px.std().item():.4f}')


def sample_grid(train, per_digit=8):
    print('\n== Samples ==')
    by_digit = {d: [] for d in range(10)}

    for i in range(len(train)):
        d = train[i][1]
        if isinstance(d, torch.Tensor):
            d = d.item()
        if len(by_digit[d]) < per_digit:
            by_digit[d].append(i)
        if all(len(v) == per_digit for v in by_digit.values()):
            break

    fig, axes = plt.subplots(10, per_digit, figsize=(8, 10))

    for digit in range(10):
        for j, i in enumerate(by_digit[digit]):
            axes[digit, j].imshow(train[i][0], cmap='gray')
            axes[digit, j].set_xticks([])
            axes[digit, j].set_yticks([])
        axes[digit, 0].set_ylabel(str(digit), rotation=0, labelpad=12, fontsize=12)

    fig.tight_layout()
    fig.savefig(ASSETS / 'samples.pdf')
    plt.close(fig)


def class_balance(train, val, test):
    print('\n== Class Balance ==')
    counts = {}

    for name, dataset in (('train', train), ('val', val), ('test', test)):
        per_digit = [0] * N_CLASSES

        for i in range(len(dataset)):
            per_digit[dataset[i][1]] += 1

        counts[name] = per_digit

    ratio = max(counts['train']) / min(counts['train'])

    for name in ('train', 'val', 'test'):
        print(f'Number of samples per digit of {name}: {counts[name]}')

    print(f'Imbalance ratio (train max/min): {ratio:.2f}x (balanced under 2.0x)')
    x = range(N_CLASSES)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for k, name in enumerate(('train', 'val', 'test')):
        ax.bar([i + (k - 1) * 0.25 for i in x], counts[name], width=0.25, label=name)

    ax.set_xlabel('digit')
    ax.set_ylabel('images')
    ax.set_xticks(list(x))
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=3)
    fig.tight_layout()
    fig.savefig(ASSETS / 'class_balance.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    train, val, test = load_splits()
    original()
    overview(train, val, test)
    split(train, val, test)
    split_plot(train, val, test)
    geometry(train, val, test)
    normalization(train)
    sample_grid(train)
    class_balance(train, val, test)


if __name__ == '__main__':
    main()
