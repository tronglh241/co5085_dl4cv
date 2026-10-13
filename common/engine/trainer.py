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
    ):
        super().__init__(data, model, device=device, max_epochs=max_epochs)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.prepare_batch = prepare_batch or default_prepare_batch
        self.should_stop = False

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        x, y = self.prepare_batch(batch, self.device)
        loss = self.loss_fn(self.model(x), y)
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item(), 'n': y.size(0)}

    def run(self):
        self._fire(Events.STARTED)

        while self.state['epoch'] < self.max_epochs and not self.should_stop:
            self._fire(Events.EPOCH_STARTED)
            total_loss = 0.0
            total_n = 0

            for batch in self.data:
                self._fire(Events.ITERATION_STARTED)
                self.output = self.train_step(batch)
                total_loss += self.output['loss'] * self.output['n']
                total_n += self.output['n']
                self.state['iteration'] += 1
                self._fire(Events.ITERATION_COMPLETED)

            self.state['epoch'] += 1
            # Sample-weighted loss of the step that was optimized, dropout included.
            # Evaluators log a separate loss with the model in eval mode.
            self.result = {'n': total_n}
            if total_n:
                self.result['loss'] = total_loss / total_n
            self._fire(Events.EPOCH_COMPLETED)

        self._fire(Events.COMPLETED)
