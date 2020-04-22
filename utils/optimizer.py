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

class MyAdam:
    def __init__(self, params, lr, betas=(0.9, 0.999)):
        self.params = params
        self.n_params = len(self.params)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.gs = [torch.zeros_like(p) for p in self.params]
        self.Gs = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is None:
                continue
            p.grad.data[:] = 0

    def step(self):
        for i in range(self.n_params):
            self.gs[i] = self.beta1 * self.gs[i] + self.params[i].grad * (1 - self.beta1)
            self.Gs[i] = self.beta2 * self.Gs[i] + self.params[i].grad.pow(2) * (1 - self.beta2)
            dp = self.lr * self.gs[i] / (self.Gs[i] + 1e-10).sqrt()
            self.params[i].data[:] -= dp




