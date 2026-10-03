import torch

from .base import Metric


class ConfusionMatrix(Metric):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

    def update(self, output):
        pred = output['y_pred'].argmax(dim=1).cpu()
        targets = output['y'].cpu().long()
        self.cm.index_put_((targets, pred), torch.ones_like(targets), accumulate=True)

    def compute(self):
        return self.cm
