import torch

class DirOpt:
    def __init__(self, params, lr, decay=0.1):
        self.decay = decay
        self.lr = lr
        self.params = params
        self.last_loss = 1e+10
        self.n_decays = 0

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.data[:] = 0.

    def step(self, loss: float):
        if loss > self.last_loss:
            self.lr *= self.decay
            self.n_decays += 1

        self.last_loss = loss

        gs = [p.grad for p in self.params]
        norm = sum((g**2).sum() for g in gs) ** 0.5
        ds = [g / norm for g in gs]

        for p, d in zip(self.params, ds):
            p.data[:] -= d * self.lr




