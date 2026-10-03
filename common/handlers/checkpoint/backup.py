from common.seed import get_rng_state

from .base import Checkpoint


class BackupCheckpoint(Checkpoint):
    def __init__(self, dirname, modules, n_saved=None):
        super().__init__(dirname, modules, n_saved)

    def filename_prefix(self):
        return 'backup'

    def state_filename(self, engine):
        return f'{self.filename_prefix()}_epoch_{engine.state["epoch"]}.pt'

    def build_state(self, engine):
        state = super().build_state(engine)
        state['rng'] = get_rng_state()
        return state
