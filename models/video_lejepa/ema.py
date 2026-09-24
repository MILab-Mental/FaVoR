from collections import OrderedDict

import torch


class ModelEMA:
    """A checkpoint/evaluation shadow copy; it is never used as a loss target."""

    def __init__(self, model, decay=0.9999, update_every=32, device=None):
        if not 0 <= decay < 1:
            raise ValueError(f"EMA decay must be in [0,1), got {decay}")
        if update_every < 1:
            raise ValueError(f"EMA update_every must be >=1, got {update_every}")
        self.decay = float(decay)
        self.update_every = int(update_every)
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.num_updates = 0
        self.last_step = 0
        self.shadow = OrderedDict()
        self._initialize(model)

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    def _initialize(self, model):
        self.shadow.clear()
        for name, value in self._unwrap(model).state_dict().items():
            if value.is_floating_point():
                self.shadow[name] = value.detach().to(self.device or value.device, torch.float32).clone()

    @torch.no_grad()
    def update(self, model, step):
        step = int(step)
        if step <= 0 or step % self.update_every:
            return False
        elapsed = max(step - self.last_step, 1)
        decay = self.decay**elapsed
        current = self._unwrap(model).state_dict()
        for name, average in self.shadow.items():
            average.mul_(decay).add_(current[name].detach().to(average), alpha=1 - decay)
        self.last_step = step
        self.num_updates += 1
        return True

    def state_dict(self):
        return {
            "shadow": OrderedDict((k, v.detach().cpu()) for k, v in self.shadow.items()),
            "decay": self.decay,
            "update_every": self.update_every,
            "num_updates": self.num_updates,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state):
        shadow = state.get("shadow", state)
        self.shadow = OrderedDict((k, v.to(self.device or "cpu")) for k, v in shadow.items())
        self.num_updates = int(state.get("num_updates", 0))
        self.last_step = int(state.get("last_step", 0))
