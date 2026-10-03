from datetime import datetime
from pathlib import Path

import torch

from common.engine.engine import Events


class Checkpoint:
    def __init__(self, dirname, modules, n_saved=None):
        self.dirname = Path(dirname, datetime.now().strftime('%y%m%d%H%M%S'))
        self.module_names = list(modules)
        self.n_saved = n_saved
        self.modules = {}
        self._saved = []
        self.dirname.mkdir(parents=True, exist_ok=True)

    def attach(self, trainer, namespace):
        for name in self.module_names:
            if not isinstance(name, str):
                raise TypeError(f'checkpoint module refs must be quoted name strings, got {name!r}')

        self.modules = {name: namespace[name] for name in self.module_names}
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self)

    def filename_prefix(self):
        raise NotImplementedError

    def state_filename(self, engine):
        raise NotImplementedError

    def save_priority(self, engine):
        return engine.state['epoch']

    def should_save(self, engine):
        return True

    def build_state(self, engine):
        state = {}

        for name, module in self.modules.items():
            state[name] = module.state_dict() if hasattr(module, 'state_dict') else module

        return state

    def prune(self):
        if self.n_saved is None:
            return

        self._saved.sort(key=lambda entry: entry['priority'])
        victims = self._saved[: max(0, len(self._saved) - self.n_saved)]
        self._saved = self._saved[len(victims):]

        for victim in victims:
            victim['path'].unlink(missing_ok=True)

    def __call__(self, engine):
        if not self.should_save(engine):
            return

        path = self.dirname / self.state_filename(engine)
        torch.save(self.build_state(engine), path)

        self._saved.append({'priority': self.save_priority(engine), 'path': path})
        self._saved.sort(key=lambda entry: entry['priority'])
        self.prune()
