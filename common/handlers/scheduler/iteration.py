from common.engine.engine import Events

from .base import Scheduler


class IterationScheduler(Scheduler):
    '''Steps the wrapped scheduler after every training iteration.'''

    def attach(self, trainer, event=Events.ITERATION_COMPLETED):
        super().attach(trainer, event)

    def __call__(self, engine):
        self.scheduler.step()
