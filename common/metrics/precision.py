import torch

from .base import Metric
from .confusion_matrix import ConfusionMatrix


def precision_from_cm(cm):
    cm = cm.float()
    col_sum = cm.sum(dim=0)
    return torch.where(col_sum > 0, cm.diag() / col_sum.clamp_min(1e-12), torch.zeros_like(col_sum))


class Precision(Metric):
    def __init__(self, num_classes, average=False):
        self.average = average
        self.cm = ConfusionMatrix(num_classes)
        self.reset()

    def reset(self):
        self.cm.reset()

    def update(self, output):
        self.cm.update(output)

    def compute(self):
        precision = precision_from_cm(self.cm.compute())
        return precision.mean().item() if self.average else precision
