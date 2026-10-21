import torch

from common.engine.engine import Events
from common.seed import set_rng_state


class CheckpointLoader:
    def __init__(self, path, modules=None, map_location='cpu'):
        self.path = path
        self.module_names = list(modules) if modules is not None else None
        self.map_location = map_location

    def attach(self, trainer, namespace):
        self.namespace = namespace
        trainer.add_event_handler(Events.STARTED, self)

    def __call__(self, engine):
        if not self.path:
            return

        checkpoint = torch.load(self.path, map_location=self.map_location, weights_only=False)

        if self.module_names is None:
            names = [key for key in checkpoint if key != 'rng']
        else:
            names = self.module_names

        for name in names:
            self.namespace[name].load_state_dict(checkpoint[name])

        if 'rng' in checkpoint:
            set_rng_state(checkpoint['rng'])
