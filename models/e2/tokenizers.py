'''Turn an image into a sequence of tokens.

Each tokenizer maps (B, C, H, W) to (B, N, D) and exposes num_tokens and embed_dim
so the classifier can build a positional embedding. A channel-as-token variant is
not included: MNIST has C = 1, so that sequence would be a single token.
'''

import torch.nn as nn


def _grid_size(image_size, kernel, stride):
    if stride <= 0 or kernel <= 0:
        raise ValueError(f'kernel {kernel} and stride {stride} must be positive')
    side = (image_size - kernel) // stride + 1
    if side <= 0:
        raise ValueError(f'kernel {kernel} stride {stride} does not fit image_size {image_size}')
    uncovered = (image_size - kernel) % stride
    if uncovered:
        raise ValueError(
            f'kernel {kernel} stride {stride} leaves {uncovered}px uncovered on image_size {image_size}'
        )
    return side


class PatchTokenizer(nn.Module):
    def __init__(self, image_size=28, patch_size=4, stride=None, in_channels=1, embed_dim=64):
        super().__init__()
        if stride is None:
            stride = patch_size
        side = _grid_size(image_size, patch_size, stride)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride)
        self.num_tokens = side * side
        self.embed_dim = embed_dim

    def forward(self, x):
        tokens = self.proj(x)
        return tokens.flatten(2).transpose(1, 2)


class RowTokenizer(nn.Module):
    def __init__(self, image_size=28, in_channels=1, embed_dim=64):
        super().__init__()
        self.image_size = image_size
        self.proj = nn.Linear(in_channels * image_size, embed_dim)
        self.num_tokens = image_size
        self.embed_dim = embed_dim

    def forward(self, x):
        batch, channels, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(f'expected image size {self.image_size}, got {(height, width)}')
        rows = x.permute(0, 2, 1, 3).reshape(batch, height, channels * width)
        return self.proj(rows)


class CNNStemTokenizer(nn.Module):
    def __init__(self, image_size=28, in_channels=1, embed_dim=64, channels=(32, 64)):
        super().__init__()
        if image_size % (2 ** len(tuple(channels))) != 0:
            raise ValueError(f'image_size {image_size} must survive {len(tuple(channels))} MaxPool2ds')

        layers = []
        fan_in = in_channels
        for width in tuple(channels):
            layers += [
                nn.Conv2d(fan_in, width, kernel_size=3, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            fan_in = width
        layers.append(nn.Conv2d(fan_in, embed_dim, kernel_size=1))
        self.stem = nn.Sequential(*layers)
        spatial = image_size // (2 ** len(tuple(channels)))
        self.num_tokens = spatial * spatial
        self.embed_dim = embed_dim

    def forward(self, x):
        feat = self.stem(x)
        return feat.flatten(2).transpose(1, 2)


class PixelTokenizer(nn.Module):
    '''One token per pixel. N = 784 on MNIST, so attention is expensive; use a GPU.'''

    def __init__(self, image_size=28, in_channels=1, embed_dim=64):
        super().__init__()
        self.image_size = image_size
        self.proj = nn.Linear(in_channels, embed_dim)
        self.num_tokens = image_size * image_size
        self.embed_dim = embed_dim

    def forward(self, x):
        batch, _, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(f'expected image size {self.image_size}, got {(height, width)}')
        pixels = x.permute(0, 2, 3, 1).reshape(batch, height * width, -1)
        return self.proj(pixels)
