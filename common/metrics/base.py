from ..engine.engine import Events


class Metric:
    def reset(self):
        raise NotImplementedError

    def update(self, output):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def attach(self, evaluator, name):
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: self.reset())
        evaluator.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: self.update(engine.output))
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: engine.result.update({name: self.compute()}))
