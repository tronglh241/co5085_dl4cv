from common.engine.trainer import Trainer


class LossTrainer(Trainer):
    '''Trainer that also reports the mean training loss of each epoch as `result`.

    The loss is the running value seen while training (dropout on, augmented images), so it is
    not comparable with a loss computed in eval mode, nor between the softmax and ArcFace heads.
    The handlers are registered here, before run_config attaches the TensorBoard logger, so the
    result is already set when the logger reads it at the end of the epoch.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_sum = 0.0
        self._loss_count = 0
        self.on_epoch_started(self._reset_loss)
        self.on_iteration_completed(self._accumulate_loss)
        self.on_epoch_completed(self._finalize_loss)

    def _reset_loss(self, engine):
        self._loss_sum = 0.0
        self._loss_count = 0

    def _accumulate_loss(self, engine):
        self._loss_sum += self.output['loss'] * self.output['n']
        self._loss_count += self.output['n']

    def _finalize_loss(self, engine):
        self.result = {'loss': self._loss_sum / max(self._loss_count, 1), 'n': self._loss_count}
