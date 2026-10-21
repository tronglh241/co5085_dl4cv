'''Download Food-101, resize it once, and write the split lists.

Decoding the original JPEGs (up to 512px) at training speed needs many CPU
cores. Resizing the shorter side to 256px once makes every later epoch cheaper;
RandomResizedCrop(224) and Resize(256) + CenterCrop(224) both still fit.

The copy keeps torchvision's layout, so datasets.food101.Food101 reads either:
    data/food-101/{images,meta}              original download (--raw data/)
    data/food101_256/food-101/{images,meta}  resized copy (--out data/food101_256)
    data/food101_256/splits/*.csv            image ids of every split

    python scripts/a1/prepare_data.py --raw data/ --out data/food101_256
'''

import argparse
import csv
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from PIL import Image
from torchvision.datasets import Food101 as _TorchvisionFood101

from datasets.food101 import Food101

# train_f10 is the subset the train-split evaluator reads every epoch.
SPLITS = (
    ('train', 1.0),
    ('train', 0.1),
    ('val', 1.0),
    ('test', 1.0),
)


def _resize(job):
    source, target, size = job
    if target.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert('RGB')
        scale = size / min(image.size)

        if scale < 1:
            new_size = (round(image.width * scale), round(image.height * scale))
            image = image.resize(new_size, Image.BICUBIC)

        image.save(target, quality=90)


def resize_all(raw_root, out_root, size, workers):
    sources = sorted((raw_root / 'images').glob('*/*.jpg'))
    jobs = [(path, out_root / 'images' / path.parent.name / path.name, size) for path in sources]

    with Pool(workers) as pool:
        for done, _ in enumerate(pool.imap_unordered(_resize, jobs, chunksize=64), start=1):
            if done % 10000 == 0 or done == len(jobs):
                print(f'resized {done}/{len(jobs)}')

    shutil.copytree(raw_root / 'meta', out_root / 'meta', dirs_exist_ok=True)


def write_splits(out_root, val_ratio, seed):
    split_dir = out_root / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)

    for split, fraction in SPLITS:
        dataset = Food101(out_root, split=split, val_ratio=val_ratio, seed=seed, fraction=fraction)
        suffix = '' if fraction == 1 else f'_f{round(fraction * 100)}'
        path = split_dir / f'{split}{suffix}.csv'

        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['id', 'label'])
            writer.writerows(zip(dataset.ids, dataset.targets))

        print(f'wrote {path} ({len(dataset)} images)')


def main():
    parser = argparse.ArgumentParser(description='Prepare Food-101 for A1.')
    parser.add_argument('--raw', default='data/', help='torchvision download root')
    parser.add_argument('--out', default='data/food101_256')
    parser.add_argument('--size', type=int, default=256, help='shorter side after resizing')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    _TorchvisionFood101(args.raw, download=True)
    raw_root = Path(args.raw) / 'food-101'
    out_root = Path(args.out)

    resize_all(raw_root, out_root / 'food-101', args.size, args.workers)
    write_splits(out_root, args.val_ratio, args.seed)


if __name__ == '__main__':
    main()
