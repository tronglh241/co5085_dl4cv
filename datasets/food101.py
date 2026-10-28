'''Food-101 with a val split and a train fraction on top of torchvision's.

torchvision.datasets.Food101 reads the official split from meta/{train,test}.json:
750 train and 250 test images per class. Like datasets/mnist.py does for MNIST,
this subclass carves a stratified val split out of train and can keep only a
fraction of train for the data-size ablation.

The test images were cleaned by hand; the train images keep their label noise
on purpose, so val (taken from train) is noisy too.

root holds food-101/{images,meta}: data/ for the original download, or
data/food101_256 for the copy scripts/a1/prepare_data.py writes after resizing
the shorter side to 256px and center-cropping to 224px.
'''

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision.datasets import Food101 as _TorchvisionFood101


def collate_batch(batch):
    '''Stack (image, label) pairs.

    torch 2.7.1's default collate, when num_workers > 0, builds the batch in
    shared memory and then resize_()s that storage. Shared memory cannot be
    resized, so the first batch raises "Trying to resize storage that is not
    resizable". torch.stack allocates a normal tensor instead.
    '''
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)


def nested_subset(indices, labels, fraction, seed):
    '''Keep `fraction` of each class.

    Each class gets one fixed permutation, so for a given seed the 10% subset is
    contained in the 25% subset, and so on.
    '''
    rng = np.random.default_rng(seed)
    kept = []

    for label in np.unique(labels[indices]):
        members = np.sort(indices[labels[indices] == label])
        order = rng.permutation(len(members))
        kept.append(members[order[:max(1, round(fraction * len(members)))]])

    return np.concatenate(kept)


class Food101(_TorchvisionFood101):
    def __init__(
        self,
        root,
        split='train',
        val_ratio=0.1,
        seed=42,
        fraction=1.0,
        **kwargs,
    ):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'unknown split: {split!r} (expected "train", "val" or "test")')

        if not 0 < fraction <= 1:
            raise ValueError(f'fraction must be in (0, 1], got {fraction}')

        if fraction < 1 and split != 'train':
            raise ValueError('fraction only subsamples the train split; val and test stay whole')

        # download=False: prepare_data.py already wrote the 224px crops. Leaving
        # this True makes a fresh container fetch the original ~5GB archive and
        # train on those variable-size images instead.
        super().__init__(root, split='test' if split == 'test' else 'train', download=False, **kwargs)

        if split == 'test':
            self.indices = None
        else:
            labels = np.array(self._labels)
            train_idx, val_idx = train_test_split(
                np.arange(len(labels)),
                test_size=val_ratio,
                stratify=labels,
                random_state=seed,
            )
            keep = train_idx if split == 'train' else val_idx

            if fraction < 1:
                keep = nested_subset(keep, labels, fraction, seed)

            # Sorted, so evaluation visits the images in a fixed order.
            self.indices = np.sort(keep).tolist()

    def _positions(self):
        return range(len(self._image_files)) if self.indices is None else self.indices

    @property
    def images_folder(self):
        return self._images_folder

    @property
    def ids(self):
        '''<class>/<image id> of every image in this split, as in the meta files.'''
        return [
            self._image_files[index].relative_to(self._images_folder).with_suffix('').as_posix()
            for index in self._positions()
        ]

    @property
    def targets(self):
        return [self._labels[index] for index in self._positions()]

    def load_image(self, index):
        '''The RGB image at `index` of this split, before any transform.'''
        if self.indices is not None:
            index = self.indices[index]

        return self.loader(self._image_files[index])

    def __len__(self):
        return len(self._image_files) if self.indices is None else len(self.indices)

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]

        return super().__getitem__(index)
