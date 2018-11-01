from .system import TDSystem
from .optimizers import Optimizer, Adam, SGD
from tqdm import tqdm, trange

class Worker:
    def __init__(self, L, c):
        self.s = TDSystem(L, c)
        self.opt = Adam([self.s.angles], lr=0.1, betas=(0.9, 0.999))

    def remake_spins(self):
        self.s.make_xy()

    def set_lr(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def optimizer(self, lr, n_steps):
        es = []
        self.set_lr(lr)
        for _ in trange(n_steps):
            self.opt.zero_grad()
            e = self.s.energy_density()
            e.backward()
            es.append(float(e))
            self.opt.step()
        return es