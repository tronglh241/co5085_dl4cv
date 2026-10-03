from .base import Scheduler


class MetricScheduler(Scheduler):
    def __init__(self, scheduler, evaluator, score_name='loss'):
        super().__init__(scheduler)
        self.evaluator = evaluator
        self.score_name = score_name

    def __call__(self, engine):
        metric = self.evaluator.result[self.score_name]

        if metric is None:
            raise ValueError(f'no metric {self.score_name!r} to step on')

        self.scheduler.step(metric)
