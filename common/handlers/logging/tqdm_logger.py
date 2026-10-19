from tqdm import tqdm

from common.engine.engine import Events


class TQDMLogger:
    def attach(self, trainer=None, evaluators=None):
        self._eval_position = 1 if trainer is not None else 0

        if trainer is not None:
            trainer.add_event_handler(Events.EPOCH_STARTED, self._train_started)
            trainer.add_event_handler(Events.ITERATION_COMPLETED, self._train_iter)
            trainer.add_event_handler(Events.EPOCH_COMPLETED, self._train_done)

        for name, evaluator in (evaluators or {}).items():
            evaluator.add_event_handler(Events.EPOCH_STARTED, self._eval_started, name)
            evaluator.add_event_handler(Events.ITERATION_COMPLETED, self._eval_iter)
            evaluator.add_event_handler(Events.EPOCH_COMPLETED, self._eval_done)

    def _train_started(self, engine):
        epoch = engine.state['epoch'] + 1
        desc = f'Epoch {epoch}/{engine.max_epochs}'
        self._train_bar = tqdm(
            total=len(engine.data),
            desc=desc,
            position=0,
            leave=False,
        )

    def _train_iter(self, engine):
        self._train_bar.update(1)

        if 'loss' in engine.output:
            self._train_bar.set_postfix(loss=f'{engine.output["loss"]:.4f}')

    def _train_done(self, engine):
        self._train_bar.close()

    def _eval_started(self, engine, name):
        self._eval_bar = tqdm(
            total=len(engine.data),
            desc=f'Eval/{name}',
            position=self._eval_position,
            leave=self._eval_position == 0,
        )

    def _eval_iter(self, engine):
        self._eval_bar.update(1)

    def _eval_done(self, engine):
        self._eval_bar.close()
