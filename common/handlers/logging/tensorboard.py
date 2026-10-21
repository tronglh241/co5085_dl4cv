import os

from torch.utils.tensorboard import SummaryWriter

from common.engine.engine import Events
from common.stamp import RUN_STAMP


class TensorBoardLogger:
    def __init__(self, log_dir):
        self.root = log_dir
        self.stamp = RUN_STAMP
        self._writer = None

    @property
    def log_dir(self):
        return os.path.join(self.root, self.stamp)

    @property
    def writer(self):
        # Opened on first use, so loading a config never leaves an empty run folder.
        if self._writer is None:
            self._writer = SummaryWriter(self.log_dir)
        return self._writer

    def use_stamp(self, stamp):
        if self._writer is not None:
            raise RuntimeError(f'writer already open at {self.log_dir}')
        self.stamp = stamp

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

            if value is None:  # Loss over an empty loader
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
        if self._writer is not None:
            self._writer.flush()
