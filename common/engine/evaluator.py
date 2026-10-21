import torch

from .engine import Engine, Events, default_prepare_batch


class Evaluator(Engine):
    def __init__(
        self,
        data,
        model,
        device=None,
        prepare_batch=None,
        amp=False,
        channels_last=False,
    ):
        super().__init__(data, model, device=device, amp=amp, channels_last=channels_last)
        self.prepare_batch = prepare_batch or default_prepare_batch

    @torch.no_grad()
    def eval_step(self, batch):
        x, y = self.prepare_batch(batch, self.device)

        with self.autocast():
            y_pred = self.model(self.to_input(x))

        # Metrics and the loss see float32 logits even under bf16 autocast.
        return {'y_pred': y_pred.float(), 'y': y}

    def run(self):
        self._fire(Events.STARTED)
        self.result = {}
        self.model.eval()
        self._fire(Events.EPOCH_STARTED)

        total_n = 0

        for batch in self.data:
            self._fire(Events.ITERATION_STARTED)

            self.output = self.eval_step(batch)
            total_n += self.output['y'].size(0)

            self.state['iteration'] += 1

            self._fire(Events.ITERATION_COMPLETED)

        self.state['epoch'] += 1
        self.result['n'] = total_n

        self._fire(Events.EPOCH_COMPLETED)
        self._fire(Events.COMPLETED)

        return self.result
