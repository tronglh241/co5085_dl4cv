from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST as _TorchvisionMNIST


class MNIST(_TorchvisionMNIST):
    def __init__(
        self,
        root,
        split='train',
        val_ratio=1 / 6,
        seed=42,
        **kwargs,
    ):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'unknown split: {split!r} (expected "train", "val" or "test")')

        super().__init__(root, train=(split != 'test'), download=True, **kwargs)

        if split == 'test':
            self.indices = None
        else:
            train_idx, val_idx = train_test_split(
                range(len(self.data)),
                test_size=val_ratio,
                stratify=self.targets.numpy(),
                random_state=seed,
            )
            self.indices = list(train_idx if split == 'train' else val_idx)

    def __len__(self):
        return len(self.data) if self.indices is None else len(self.indices)

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]

        return super().__getitem__(index)
