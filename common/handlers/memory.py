'''Peak GPU memory of the training iterations in one epoch.

Like EpochTimer, the reading is taken on the trainer's last
ITERATION_COMPLETED, so the evaluators that run on EPOCH_COMPLETED do not
count. Logged as training/peak_mem_gb; nothing is logged on CPU.
'''

import torch

from common.engine.engine import Events


class PeakMemory:
    def __init__(self, logger=None):
        self.logger = logger
        self._peak = None

    def attach(self, trainer):
        if trainer.device.type != 'cuda':
            return

        trainer.add_event_handler(Events.EPOCH_STARTED, self._on_start)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self._on_iteration)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self._on_epoch)

    def _on_start(self, engine):
        torch.cuda.reset_peak_memory_stats(engine.device)
        self._peak = None

    def _on_iteration(self, engine):
        self._peak = torch.cuda.max_memory_allocated(engine.device)

    def _on_epoch(self, engine):
        if self._peak is None or self.logger is None:
            return

        self.logger.writer.add_scalar('training/peak_mem_gb', self._peak / 2**30, engine.state['epoch'])
