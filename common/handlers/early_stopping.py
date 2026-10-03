from common.engine.engine import Events


class EarlyStopping:
    def __init__(
        self,
        patience,
        evaluator,
        mode='min',
        score_name='loss',
    ):
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.patience = patience
        self.evaluator = evaluator
        self.mode = mode
        self.score_name = score_name
        self.best = None
        self.bad_epochs = 0

    def is_better(self, metric):
        if self.best is None:
            return True

        if self.mode == 'min':
            return metric < self.best

        return metric > self.best

    def update(self, metric):
        if metric is None:
            return False

        if self.is_better(metric):
            self.best = metric
            self.bad_epochs = 0

            return False

        self.bad_epochs += 1

        return self.bad_epochs > self.patience

    def attach(self, trainer):
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self)

    def __call__(self, engine):
        if self.update(self.evaluator.result[self.score_name]):
            engine.should_stop = True

    def state_dict(self):
        return {'best': self.best, 'bad_epochs': self.bad_epochs}

    def load_state_dict(self, state_dict):
        for key in ('best', 'bad_epochs'):
            if key not in state_dict:
                raise ValueError(f'missing required key {key!r} in {sorted(state_dict)}')

            setattr(self, key, state_dict[key])
