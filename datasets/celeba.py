import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def collate_batch(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)


def collate_pairs(batch):
    index_a, index_b, labels = zip(*batch)
    return torch.tensor(index_a), torch.tensor(index_b), torch.tensor(labels)


def read_rows(path, required):
    path = Path(path)
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        return list(reader)


class CelebA(Dataset):
    def __init__(self, root, split='train', transform=None):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'unknown split: {split!r} (expected "train", "val" or "test")')

        self.root = Path(root)
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
    def __init__(self, root, split='val', pairs=None):
        if split not in ('val', 'test'):
            raise ValueError(f'unknown pair split: {split!r} (expected "val" or "test")')

        self.root = Path(root)
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
