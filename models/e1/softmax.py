import torch.nn as nn


class SoftmaxClassifier(nn.Module):
    def __init__(
        self,
        in_channels=1,
        image_size=28,
        num_classes=10,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(in_channels * image_size * image_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)
