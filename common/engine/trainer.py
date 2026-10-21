import torch

from .engine import Engine, Events, default_prepare_batch


class Trainer(Engine):
    def __init__(
        self,
        data,
        model,
        optimizer,
        loss_fn,
        device=None,
        max_epochs=1,
        prepare_batch=None,
        amp=False,
        channels_last=False,
        clip_grad_norm=None,
    ):
        super().__init__(
            data,
            model,
            device=device,
            max_epochs=max_epochs,
            amp=amp,
            channels_last=channels_last,
        )
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.prepare_batch = prepare_batch or default_prepare_batch
        self.clip_grad_norm = clip_grad_norm
        self.should_stop = False

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        x, y = self.prepare_batch(batch, self.device)

        with self.autocast():
            loss = self.loss_fn(self.model(self.to_input(x)), y)

        loss.backward()

        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        self.optimizer.step()
        return {'loss': loss.item(), 'n': y.size(0)}

    def run(self):
        self._fire(Events.STARTED)

        while self.state['epoch'] < self.max_epochs and not self.should_stop:
            self._fire(Events.EPOCH_STARTED)

            for batch in self.data:
                self._fire(Events.ITERATION_STARTED)
                self.output = self.train_step(batch)
                self.state['iteration'] += 1
                self._fire(Events.ITERATION_COMPLETED)

            self.state['epoch'] += 1
            self._fire(Events.EPOCH_COMPLETED)

        self._fire(Events.COMPLETED)
