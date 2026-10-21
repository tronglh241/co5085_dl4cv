import os
from pathlib import Path

import torch

from common.engine.engine import Events
from common.stamp import RUN_STAMP


class Checkpoint:
    def __init__(self, dirname, modules, n_saved=None):
        self.root = Path(dirname)
        self.stamp = RUN_STAMP
        self.module_names = list(modules)
        self.n_saved = n_saved
        self.modules = {}
        self._saved = []

    @property
    def dirname(self):
        return self.root / self.stamp

    def use_stamp(self, stamp):
        self.stamp = stamp

    def attach(self, trainer, namespace):
        for name in self.module_names:
            if not isinstance(name, str):
                raise TypeError(f'checkpoint module refs must be quoted name strings, got {name!r}')

        self.modules = {name: namespace[name] for name in self.module_names}
        # A resumed run writes into its old folder: count the files already
        # there so n_saved still prunes them. A *.tmp is a save that a
        # preemption cut short; it is never a checkpoint.
        for partial in self.dirname.glob('*.tmp'):
            partial.unlink(missing_ok=True)
        self._saved = self.existing()
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self)

    def filename_prefix(self):
        raise NotImplementedError

    def state_filename(self, engine):
        raise NotImplementedError

    def priority_from_filename(self, filename):
        '''save_priority of a file written by state_filename, or None if it does not parse.'''
        raise NotImplementedError

    def existing(self):
        saved = []

        for path in self.dirname.glob(f'{self.filename_prefix()}_*.pt'):
            priority = self.priority_from_filename(path.name)

            if priority is not None:
                saved.append({'priority': priority, 'path': path})

        saved.sort(key=lambda entry: entry['priority'])
        return saved

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

        # Created on the first save, so loading a config never leaves an empty run folder.
        self.dirname.mkdir(parents=True, exist_ok=True)
        path = self.dirname / self.state_filename(engine)
        # Written under another name and renamed: the rename is atomic, so a
        # *.pt that exists is always complete, even if the process is killed mid-save.
        partial = path.with_name(path.name + '.tmp')
        torch.save(self.build_state(engine), partial)
        os.replace(partial, path)

        self._saved.append({'priority': self.save_priority(engine), 'path': path})
        self._saved.sort(key=lambda entry: entry['priority'])
        self.prune()
