import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(
        self,
        num_classes=10,
        image_size=28,
        in_channels=1,
        hidden_dims=(256, 128),
        dropout=0.0,
    ):
        super().__init__()
        dims = (in_channels * image_size * image_size,) + tuple(hidden_dims)
        layers = [nn.Flatten(1)]

        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(fan_in, fan_out), nn.ReLU(inplace=True)]

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
