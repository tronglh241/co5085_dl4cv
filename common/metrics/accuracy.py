from .base import Metric
from .confusion_matrix import ConfusionMatrix


def accuracy_from_cm(cm):
    total = cm.sum().item()
    return cm.diag().sum().item() / total if total else 0.0


class Accuracy(Metric):
    def __init__(self, num_classes):
        self.cm = ConfusionMatrix(num_classes)
        self.reset()

    def reset(self):
        self.cm.reset()

    def update(self, output):
        self.cm.update(output)

    def compute(self):
        return accuracy_from_cm(self.cm.compute())
