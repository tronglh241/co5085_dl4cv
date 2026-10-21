import time

from common.engine.engine import Events


class EpochTimer:
    def __init__(self, logger=None):
        self.logger = logger
        self._start = None
        self._last = None

    def attach(self, trainer):
        trainer.add_event_handler(Events.EPOCH_STARTED, self._on_start)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self._on_iteration)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self._on_epoch)

    def _on_start(self, engine):
        now = time.perf_counter()
        self._start = now
        self._last = now

    def _on_iteration(self, engine):
        self._last = time.perf_counter()

    def _on_epoch(self, engine):
        if self._start is None or self._last is None or self.logger is None:
            return

        elapsed = self._last - self._start
        self.logger.writer.add_scalar('training/epoch_time', elapsed, engine.state['epoch'])
