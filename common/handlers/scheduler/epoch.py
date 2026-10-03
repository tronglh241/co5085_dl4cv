from .base import Scheduler


class EpochScheduler(Scheduler):
    def __call__(self, engine):
        self.scheduler.step()
