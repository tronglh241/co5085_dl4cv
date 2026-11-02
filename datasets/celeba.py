import csv
import shutil
import tempfile
from pathlib import Path

import gdown
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import extract_archive

CELEBA_112_URL = 'https://drive.google.com/file/d/1Mw10GLhCDVAqPb2GPE7zjMpeJxKrt1l1'
CELEBA_112_ARCHIVE = 'celeba_112.zip'
CELEBA_112_MARKERS = (
    'splits/train.csv',
    'splits/val.csv',
    'splits/test.csv',
    'pairs/val.csv',
    'pairs/test.csv',
    'identification/test_gallery.csv',
    'identification/test_probe.csv',
    'images',
)


def _is_prepared(root):
    return all((root / marker).exists() for marker in CELEBA_112_MARKERS)


def ensure_celeba_112(root, download=True):
    root = Path(root)
    if _is_prepared(root):
        return root

    if not download:
        raise FileNotFoundError(f'{root} is missing prepared CelebA files; pass download=True')

    root.parent.mkdir(parents=True, exist_ok=True)
    archive = root.parent / CELEBA_112_ARCHIVE
    if not archive.is_file():
        gdown.download(CELEBA_112_URL, str(archive), quiet=False, resume=True, use_cookies=False)
    if not archive.is_file():
        raise RuntimeError(f'download did not create {archive}')

    with tempfile.TemporaryDirectory(dir=root.parent) as temporary:
        extract_archive(str(archive), temporary)
        source = Path(temporary) / 'celeba_112'
        if not source.is_dir():
            raise RuntimeError(f'{archive} does not contain celeba_112/')
        if root.exists():
            shutil.copytree(source, root, dirs_exist_ok=True)
        else:
            shutil.move(str(source), str(root))

    missing = [marker for marker in CELEBA_112_MARKERS if not (root / marker).exists()]
    if missing:
        raise RuntimeError(f'{root} is incomplete after extraction: missing {missing}')
    return root


def collate_batch(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)


def read_rows(path, required):
    path = Path(path)
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        return list(reader)


class CelebA(Dataset):
    def __init__(self, root, split='train', transform=None, download=True):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'unknown split: {split!r} (expected "train", "val" or "test")')

        self.root = ensure_celeba_112(root, download=download)
        self.split = split
        self.transform = transform
        self.images_folder = self.root / 'images'

        rows = read_rows(self.root / 'splits' / f'{split}.csv', ('id', 'identity', 'label'))
        self.ids = [row['id'] for row in rows]
        self.identities = [int(row['identity']) for row in rows]
        self.labels = [int(row['label']) for row in rows]

    def __len__(self):
        return len(self.ids)

    def load_image(self, index):
        with Image.open(self.images_folder / self.ids[index]) as image:
            return image.convert('RGB')

    def __getitem__(self, index):
        image = self.load_image(index)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


class VerificationPairs(Dataset):
    def __init__(self, root, split='val', pairs=None, download=True):
        if split not in ('val', 'test'):
            raise ValueError(f'unknown pair split: {split!r} (expected "val" or "test")')

        self.root = ensure_celeba_112(root, download=download)
        self.split = split
        pairs_path = Path(pairs) if pairs is not None else self.root / 'pairs' / f'{split}.csv'

        split_rows = read_rows(self.root / 'splits' / f'{split}.csv', ('id', 'identity', 'label'))
        index = {row['id']: position for position, row in enumerate(split_rows)}
        pair_rows = read_rows(pairs_path, ('id_a', 'id_b', 'label'))

        try:
            self.index_a = [index[row['id_a']] for row in pair_rows]
            self.index_b = [index[row['id_b']] for row in pair_rows]
        except KeyError as error:
            raise ValueError(f'{pairs_path}: image {error.args[0]!r} is not in split {split!r}') from None

        self.labels = [int(row['label']) for row in pair_rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.index_a[index], self.index_b[index], self.labels[index]
