from contextlib import nullcontext
from enum import Enum

import torch


class Events(str, Enum):
    STARTED = 'started'
    EPOCH_STARTED = 'epoch_started'
    ITERATION_STARTED = 'iteration_started'
    ITERATION_COMPLETED = 'iteration_completed'
    EPOCH_COMPLETED = 'epoch_completed'
    COMPLETED = 'completed'


ALL_EVENTS = tuple(Events)


def default_prepare_batch(batch, device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


class Engine:
    def __init__(self, data, model, device=None, max_epochs=1, amp=False, channels_last=False):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.data = data
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.max_epochs = max_epochs
        # bf16 autocast needs no GradScaler. It is a no-op off CUDA, so the
        # same config also runs a CPU smoke test.
        self.amp = amp and self.device.type == 'cuda'
        self.channels_last = channels_last

        if channels_last:
            self.model.to(memory_format=torch.channels_last)

        self.state = {'epoch': 0, 'iteration': 0}
        self.output = None
        self.result = None
        self.handlers = {event: [] for event in ALL_EVENTS}

    def add_event_handler(self, event, handler, *args, **kwargs):
        try:
            key = Events(event)
        except ValueError:
            raise ValueError(f'unknown event {event!r}; expected one of {sorted(self.handlers)}') from None

        self.handlers[key].append((handler, args, kwargs))

        return handler

    def on_started(self, handler, *args, **kwargs):
        return self.add_event_handler(Events.STARTED, handler, *args, **kwargs)

    def on_epoch_started(self, handler, *args, **kwargs):
        return self.add_event_handler(Events.EPOCH_STARTED, handler, *args, **kwargs)

    def on_iteration_started(self, handler, *args, **kwargs):
        return self.add_event_handler(Events.ITERATION_STARTED, handler, *args, **kwargs)

    def on_iteration_completed(self, handler, *args, **kwargs):
        return self.add_event_handler(Events.ITERATION_COMPLETED, handler, *args, **kwargs)

    def on_epoch_completed(self, handler, *args, **kwargs):
        return self.add_event_handler(Events.EPOCH_COMPLETED, handler, *args, **kwargs)

    def on_completed(self, handler, *args, **kwargs):
        return self.add_event_handler(Events.COMPLETED, handler, *args, **kwargs)

    def autocast(self):
        if not self.amp:
            return nullcontext()

        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    def to_input(self, x):
        if self.channels_last and x.dim() == 4:
            return x.contiguous(memory_format=torch.channels_last)

        return x

    def _fire(self, event):
        for handler, args, kwargs in self.handlers[event]:
            handler(self, *args, **kwargs)

    def state_dict(self):
        return {'epoch': self.state['epoch'], 'iteration': self.state['iteration']}

    def load_state_dict(self, state_dict):
        for key in ('epoch', 'iteration'):
            if key not in state_dict:
                raise ValueError(f'missing required key {key!r} in {sorted(state_dict)}')

            self.state[key] = state_dict[key]
