import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import gdown
import numpy as np
from PIL import Image
from torchvision.datasets import CelebA as _TorchvisionCelebA
from torchvision.datasets.utils import extract_archive

SPLITS = ('train', 'val', 'test')
SPLIT_IDS = {0: 'train', 1: 'val', 2: 'test'}
CELEBA_URL = 'https://drive.google.com/drive/folders/1Lu7p9CmTKLvyQZEkXRuKlyC_6it7ql8S'
CELEBA_METADATA = (
    'identity_CelebA.txt',
    'list_attr_celeba.txt',
    'list_bbox_celeba.txt',
    'list_eval_partition.txt',
    'list_landmarks_align_celeba.txt',
)
TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float64)


def _read_partition(path):
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    return {name: int(split) for name, split in rows}


def _download_celeba(raw_root):
    celeba_dir = raw_root / 'celeba'
    images_dir = celeba_dir / 'img_align_celeba'
    archive = celeba_dir / 'img_align_celeba.zip'
    missing = [name for name in CELEBA_METADATA if not (celeba_dir / name).exists()]
    if missing or (not images_dir.exists() and not archive.exists()):
        celeba_dir.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(
            CELEBA_URL,
            output=str(celeba_dir),
            quiet=False,
            use_cookies=False,
            resume=True,
        )
    if not images_dir.exists():
        if not archive.exists():
            raise SystemExit(f'{archive} is missing after download')
        extract_archive(str(archive), str(celeba_dir))


def _load_metadata(raw_root, download):
    if download:
        _download_celeba(raw_root)
    dataset = _TorchvisionCelebA(
        raw_root,
        split='all',
        target_type=['identity', 'landmarks'],
        download=False,
    )
    names = [Path(name).name for name in dataset.filename]
    identities = {name: int(row[0]) for name, row in zip(names, dataset.identity.tolist())}
    landmarks = {
        name: np.asarray(row, dtype=np.float64).reshape(5, 2)
        for name, row in zip(names, dataset.landmarks_align.tolist())
    }
    partition = _read_partition(raw_root / 'celeba' / 'list_eval_partition.txt')

    missing = sorted(set(names) - set(partition))
    if missing:
        raise SystemExit(f'{len(missing)} images are missing from list_eval_partition.txt')

    return names, identities, landmarks, partition


def _identity_overlaps(rows_by_split):
    overlaps = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            common = set(rows_by_split[left]['identities']) & set(rows_by_split[right]['identities'])
            if common:
                overlaps[f'{left}/{right}'] = len(common)
    return overlaps


def _write_splits(out_root, rows_by_split, label_map):
    split_dir = out_root / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        path = split_dir / f'{split}.csv'
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['id', 'identity', 'label'])
            for row in rows_by_split[split]['rows']:
                label = label_map.get(row['identity'], -1)
                writer.writerow([row['id'], row['identity'], label])
        print(f'wrote {path} ({len(rows_by_split[split]["rows"])} images)')


def _similarity_coeffs(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rows, values = [], []

    for (x, y), (u, v) in zip(source, target):
        rows.append([x, -y, 1.0, 0.0])
        values.append(u)
        rows.append([y, x, 0.0, 1.0])
        values.append(v)

    a, b, tx, ty = np.linalg.lstsq(np.asarray(rows), np.asarray(values), rcond=None)[0]
    norm = a * a + b * b
    a, b = a / norm, b / norm
    return a, b, -(a * tx + b * ty), b * tx - a * ty


def _align(job):
    source, target, landmarks = job
    if target.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    a, b, c, d = _similarity_coeffs(landmarks, TEMPLATE)
    coeffs = (a, b, c, -b, a, d)
    with Image.open(source) as image:
        image = image.convert('RGB').transform(
            (112, 112),
            Image.AFFINE,
            coeffs,
            resample=Image.BICUBIC,
        )
        image.save(target, quality=95)


def _align_all(raw_root, out_root, names, landmarks):
    jobs = [
        (
            raw_root / 'celeba' / 'img_align_celeba' / name,
            out_root / 'images' / name,
            landmarks[name],
        )
        for name in names
    ]

    for index, job in enumerate(jobs, start=1):
        _align(job)
        if index % 10000 == 0 or index == len(jobs):
            print(f'aligned {index}/{len(jobs)}')


def _write_pairs(out_root, split, rows, count, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row['identity']].append(row['id'])

    identities = sorted(groups)
    eligible = [identity for identity in identities if len(groups[identity]) >= 2]
    if len(identities) < 2 or not eligible:
        raise SystemExit(f'{split}: need at least two identities and one identity with two images')

    rng = np.random.default_rng(seed)
    seen, pairs = set(), []

    while len(pairs) < count:
        positive = bool(rng.integers(0, 2))
        if positive:
            identity = eligible[int(rng.integers(len(eligible)))]
            id_a, id_b = (str(value) for value in rng.choice(groups[identity], size=2, replace=False))
        else:
            left, right = (identities[int(value)] for value in rng.choice(len(identities), size=2, replace=False))
            id_a = str(rng.choice(groups[left]))
            id_b = str(rng.choice(groups[right]))

        key = (*sorted((id_a, id_b)), positive)
        if id_a == id_b or key in seen:
            continue

        seen.add(key)
        pairs.append((id_a, id_b, int(positive)))

    path = out_root / 'pairs' / f'{split}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id_a', 'id_b', 'label'])
        writer.writerows(pairs)
    print(f'wrote {path} ({len(pairs)} pairs)')
    return len(pairs)


def _write_identification(out_root, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['identity']].append(row['id'])

    gallery, probe = [], []
    for identity in sorted(groups):
        members = sorted(groups[identity])
        if len(members) < 2:
            continue
        gallery.append((members[0], identity))
        probe.extend((image_id, identity) for image_id in members[1:])

    folder = out_root / 'identification'
    folder.mkdir(parents=True, exist_ok=True)
    for name, entries in (('test_gallery', gallery), ('test_probe', probe)):
        path = folder / f'{name}.csv'
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['id', 'identity'])
            writer.writerows(entries)
        print(f'wrote {path} ({len(entries)} images)')

    return {'gallery': len(gallery), 'probes': len(probe)}


def _write_summary(out_root, raw_root, rows_by_split, pair_counts, identification, seed):
    images = ' '.join(f'{split}={len(rows_by_split[split]["rows"])}' for split in SPLITS)
    identities = ' '.join(
        f'{split}={len(rows_by_split[split]["identities"])}' for split in SPLITS
    )
    pairs = ' '.join(f'{split}={pair_counts[split]}' for split in ('val', 'test'))
    lines = [
        f'raw: {raw_root}',
        f'out: {out_root}',
        f'images: {images}',
        f'identities: {identities}',
        'identity overlap: none',
        f'verification pairs: {pairs}',
        f'identification: gallery={identification["gallery"]} probes={identification["probes"]}',
        f'seed: {seed}',
        'template:',
    ]
    lines.extend(f'  {x:.4f} {y:.4f}' for x, y in TEMPLATE)

    path = out_root / 'summary.txt'
    path.write_text('\n'.join(lines) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser(description='Prepare CelebA for A2.')
    parser.add_argument('--raw', default='data/celeba', help='torchvision CelebA root')
    parser.add_argument('--out', default='data/celeba_112')
    parser.add_argument(
        '--download',
        action='store_true',
        help='download CelebA from the shared Google Drive folder',
    )
    parser.add_argument('--pairs', type=int, default=10000, help='verification pairs per split')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    raw_root = Path(args.raw)
    out_root = Path(args.out)
    partition_path = raw_root / 'celeba' / 'list_eval_partition.txt'

    if not partition_path.exists() and not args.download:
        parser.error(f'{partition_path} is missing; pass --download or point --raw at an existing CelebA root')

    names, identities, landmarks, partition = _load_metadata(raw_root, args.download)

    rows_by_split = {split: {'rows': [], 'identities': set()} for split in SPLITS}
    for name in sorted(names):
        split = SPLIT_IDS[partition[name]]
        identity = identities[name]
        rows_by_split[split]['rows'].append({'id': name, 'identity': identity})
        rows_by_split[split]['identities'].add(identity)

    overlaps = _identity_overlaps(rows_by_split)
    if overlaps:
        raise SystemExit(f'identity overlap across splits: {overlaps}')

    label_map = {identity: index for index, identity in enumerate(sorted(rows_by_split['train']['identities']))}
    out_root.mkdir(parents=True, exist_ok=True)
    stale_summary = out_root / 'summary.json'
    if stale_summary.exists():
        stale_summary.unlink()

    _write_splits(out_root, rows_by_split, label_map)
    _align_all(raw_root, out_root, names, landmarks)
    pair_counts = {
        split: _write_pairs(out_root, split, rows_by_split[split]['rows'], args.pairs, args.seed)
        for split in ('val', 'test')
    }
    identification = _write_identification(out_root, rows_by_split['test']['rows'])
    path = _write_summary(out_root, raw_root, rows_by_split, pair_counts, identification, args.seed)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
