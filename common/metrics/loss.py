from .base import Metric


class Loss(Metric):
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, output):
        loss = self.loss_fn(output['y_pred'], output['y'])
        n = output['y'].size(0)
        self.total += loss.item() * n
        self.count += n

    def compute(self):
        return self.total / self.count if self.count else None
