import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(
        self,
        in_channels=1,
        channels=(16, 32),
        fc_dim=128,
        num_classes=10,
        image_size=28,
    ):
        super().__init__()
        channels = tuple(channels)
        depth = len(channels)
        assert image_size % (2**depth) == 0, 'image_size must survive depth MaxPool2ds'

        blocks = []
        fan_in = in_channels

        for width in channels:
            blocks += [
                nn.Conv2d(fan_in, width, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            fan_in = width

        self.features = nn.Sequential(*blocks)
        spatial = image_size // (2**depth)
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(channels[-1] * spatial * spatial, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
