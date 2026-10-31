import torch.nn as nn


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        num_classes=10,
        image_size=28,
        in_channels=1,
        sequence='row',
        cell='lstm',
        hidden_size=128,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
    ):
        super().__init__()

        if sequence not in ('row', 'column'):
            raise ValueError(f"sequence must be 'row' or 'column', got {sequence!r}")

        if cell not in ('lstm', 'gru'):
            raise ValueError(f"cell must be 'lstm' or 'gru', got {cell!r}")

        if num_layers < 1:
            raise ValueError(f'num_layers must be positive, got {num_layers}')

        self.image_size = image_size
        self.in_channels = in_channels
        self.sequence = sequence
        self.bidirectional = bidirectional

        recurrent = nn.LSTM if cell == 'lstm' else nn.GRU
        recurrent_dropout = dropout if num_layers > 1 else 0.0

        self.rnn = recurrent(
            input_size=in_channels * image_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes),
        )

    def forward(self, x):
        batch, channels, height, width = x.shape

        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f'expected image size {self.image_size}, got {(height, width)}'
            )

        if channels != self.in_channels:
            raise ValueError(f'expected {self.in_channels} channels, got {channels}')

        if self.sequence == 'row':
            tokens = x.permute(0, 2, 1, 3).reshape(batch, height, channels * width)
        else:
            tokens = x.permute(0, 3, 1, 2).reshape(batch, width, channels * height)

        output, _ = self.rnn(tokens)
        return self.classifier(output[:, -1])
