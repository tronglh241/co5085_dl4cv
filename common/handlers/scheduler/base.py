from common.engine.engine import Events

_NON_SERIALIZABLE_STATE = ('_reduce_lr',)


class Scheduler:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def attach(self, trainer, event=Events.EPOCH_COMPLETED):
        trainer.add_event_handler(event, self)

    def __call__(self, engine):
        raise NotImplementedError

    def state_dict(self):
        return {k: v for k, v in self.scheduler.state_dict().items() if k not in _NON_SERIALIZABLE_STATE}

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)
