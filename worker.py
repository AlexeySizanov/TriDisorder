from .system import TDSystem
from .optimizers import Optimizer, Adam, SGD
from tqdm import tqdm, trange
from .storage import Storage

class Worker:
    def __init__(self, L, c, name=None, field=None, cuda=True):
        self.s = TDSystem(L, c, field=field, cuda=cuda)
        self.opt = Adam([self.s.angles], lr=0.1, betas=(0.99, 0.99))
        # self.opt = SGD([self.s.angles], lr=0.1)
        self.name = name
        self.storage = Storage(self.name) if name is not None else None

    def remake_spins(self):
        self.s.make_spins()

    def set_lr(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def optimizer(self, lr, n_steps, progress=True):
        es = []
        self.set_lr(lr)

        iterator = range(n_steps)
        if progress == True or progress == 'tqdm':
            iterator = tqdm(iterator)

        elif progress != False:
            iterator = progress(iterator)


        for _ in iterator:
            self.opt.zero_grad()
            e = self.s.energy_density()
            e.backward()
            self.s.angles.grad[self.s.hole_inds] = 0.
            es.append(float(e))
            self.opt.step()
        return es


    def save_state(self):
        if self.storage is not None:
            self.storage.save_state(s=self.s)
        else:
            raise Exception('Worker.save_state() : storage is None')
