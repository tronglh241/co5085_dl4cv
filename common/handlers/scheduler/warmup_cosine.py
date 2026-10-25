import math

from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineLR(LRScheduler):
    '''Linear warmup, then cosine decay to min_lr_ratio * base lr.

    Counted in optimizer steps: drive it with IterationScheduler and give
    warmup_steps / total_steps as epochs * len(train loader).
    '''

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.0, last_epoch=-1):
        if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
            raise ValueError(f'need 0 <= warmup_steps < total_steps, got {warmup_steps} and {total_steps}')

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def factor(self, step):
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps

        progress = min(1.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    def get_lr(self):
        factor = self.factor(self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]
