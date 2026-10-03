import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from common.engine.engine import Events


class TensorBoardLogger:
    def __init__(self, log_dir):
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%y%m%d%H%M%S'))
        self.writer = SummaryWriter(self.log_dir)

    def attach(self, engine, prefix='', optimizer=None, epoch_source=None):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._log_result, prefix, epoch_source)

        if optimizer is not None:
            engine.add_event_handler(Events.EPOCH_COMPLETED, self._log_lr, optimizer, prefix, epoch_source)

        engine.add_event_handler(Events.COMPLETED, self._flush)

    def _epoch_of(self, engine, epoch_source):
        source = epoch_source if epoch_source is not None else engine
        return source.state['epoch']

    def _log_result(self, engine, prefix, epoch_source):
        step = self._epoch_of(engine, epoch_source)

        for metric, value in (engine.result or {}).items():
            if metric == 'n':  # Number of samples
                continue

            if hasattr(value, 'numel') and value.numel() > 1:
                for i, v in enumerate(value):
                    tag = f'{prefix}/{metric}/{i}'
                    self.writer.add_scalar(tag, float(v), step)
            else:
                tag = f'{prefix}/{metric}'
                self.writer.add_scalar(tag, float(value), step)

    def _log_lr(self, engine, optimizer, prefix, epoch_source):
        tag = f'{prefix}/lr' if prefix else 'lr'
        step = self._epoch_of(engine, epoch_source)
        groups = optimizer.param_groups

        if len(groups) == 1:
            self.writer.add_scalar(tag, groups[0]['lr'], step)
        else:
            for i, pg in enumerate(groups):
                self.writer.add_scalar(f'{tag}/group_{i}', pg['lr'], step)

    def _flush(self, engine):
        self.writer.flush()
