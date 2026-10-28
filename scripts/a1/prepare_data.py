'''Download Food-101, write the fixed 224px crop once, and write the split lists.

Decoding the original JPEGs (up to 512px) at training speed needs many CPU
cores. Each image is converted to RGB, resized so its shorter side is 256px
(up or down), center-cropped to 224px, and saved as JPEG. Train later only
augments that crop; validation and test only normalize it.

An image that already exists is left as-is. Delete the prepared images before
re-running when the crop changes, or the old files stay in place.

The copy keeps torchvision's layout, so datasets.food101.Food101 reads either:
    data/food-101/{images,meta}              original download (--raw data/)
    data/food101_256/food-101/{images,meta}  224px center crops (--out data/food101_256)
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


def _resize_shorter_side(image, size):
    '''Resize so the shorter side is `size`, scaling up as well as down.'''
    width, height = image.size
    short = min(width, height)
    if short == size:
        return image
    scale = size / short
    return image.resize((round(width * scale), round(height * scale)), Image.BICUBIC)


def _center_crop(image, crop):
    width, height = image.size
    if width < crop or height < crop:
        raise ValueError(f'cannot center-crop {crop}px from {width}x{height}')
    left = (width - crop) // 2
    top = (height - crop) // 2
    return image.crop((left, top, left + crop, top + crop))


def _prepare(job):
    source, target, size, crop = job
    if target.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert('RGB')
        image = _center_crop(_resize_shorter_side(image, size), crop)
        image.save(target, quality=90)


def resize_all(raw_root, out_root, size, crop, workers):
    sources = sorted((raw_root / 'images').glob('*/*.jpg'))
    jobs = [
        (path, out_root / 'images' / path.parent.name / path.name, size, crop)
        for path in sources
    ]

    with Pool(workers) as pool:
        for done, _ in enumerate(pool.imap_unordered(_prepare, jobs, chunksize=64), start=1):
            if done % 10000 == 0 or done == len(jobs):
                print(f'prepared {done}/{len(jobs)}')

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
    parser.add_argument('--size', type=int, default=256, help='shorter side before the center crop')
    parser.add_argument('--crop', type=int, default=224, help='center-crop side saved to disk')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.size < args.crop:
        parser.error(f'--size must be at least --crop, got {args.size} and {args.crop}')

    _TorchvisionFood101(args.raw, download=True)
    raw_root = Path(args.raw) / 'food-101'
    out_root = Path(args.out)

    resize_all(raw_root, out_root / 'food-101', args.size, args.crop, args.workers)
    write_splits(out_root, args.val_ratio, args.seed)


if __name__ == '__main__':
    main()
