import gc
from itertools import count
from typing import Optional
import pickle

import numpy as np
import torch
from tqdm import tqdm

from system3d import TDSystem3D

class Worker3D:
    def __init__(self, L, H, c):
        self.H = H
        self.c = c
        self.L = L

        self.n_realizations = 0
        self.dH1 = []
        self.dH2 = []
        self.chirality = []
        self.fourier = np.zeros((L, L, H))

    def _make_one_realization(self, lr=0.5, n_steps=3000, device='cpu', progress=False):
        s = TDSystem3D(L=self.L, H=self.H, c=self.c, device=device)
        s.save_init_state()

        for i in count(1):
            if i == 100:
                return False
            s.optimize_em(n_steps=n_steps, lr=lr, progress=progress)
            gc.collect()
            qe, deq, max_angle, mean_angle  = s.check_minimum()
            if progress:
                print(f'\nE = {qe:.7f}, eq_diff = {deq:.10f},  max angle = {max_angle:.10f},  mean_angle = {mean_angle:.10f}')
            if mean_angle > 1.:
                lr *= 0.8
                print(f'mean angle > 1.: lr -> {lr}')
                s.restore_init_state()
                continue
            if max_angle < 0.001:
                break

        vals, vecs, dH1 = s.find_twist()
        self.dH1.append(np.linalg.norm(dH1))
        self.dH2.append(vals)

        chirality = s.get_chirality()
        self.chirality.append(chirality)

        fourier = s.get_fourier().cpu()
        self.fourier = (self.fourier * self.n_realizations + fourier.numpy()) / (self.n_realizations + 1)

        self.n_realizations += 1
        del s
        gc.collect()

        return True

    def save(self, filename):
        np.savez(file=filename, L=self.L, H=self.H, c=self.c, n_realizations=self.n_realizations,
                 dH1=self.dH1, dH2=self.dH2, chirality=self.chirality, fourier=self.fourier)

    @classmethod
    def load(cls, filename):
        f = np.load(filename + '.npz')
        new = Worker3D(L=int(f['L']), H=int(f['L']), c=float(f['c']))
        new.n_realizations = int(f['n_realizations'])
        new.dH1 = [*f['dH1']]
        new.dH2 = [*f['dH2']]
        new.chirality = [*f['chirality']]
        new.fourier = f['fourier']
        return new

    def __add__(self, other: 'Worker3D') -> 'Worker3D':
        if self.H != other.H or self.L != other.L or self.c != other.c:
            raise ValueError('Both workers must have same dimensions and concentration.')

        new: Worker3D = Worker3D(H=self.H, L=self.L, c=self.c)
        new.n_realizations = self.n_realizations + other.n_realizations
        new.dH1 = self.dH1 + other.dH1
        new.dH2 = self.dH2 + other.dH2
        new.chirality = self.chirality + other.chirality
        new.fourier = (self.n_realizations * self.fourier + other.n_realizations * other.fourier) / new.n_realizations
        return new

    def do(self, n: Optional[int] = None, save_every: Optional[int] = None,
           device='cpu', folder=None, prefix=None):

        if n is None and save_every is None:
            raise Exception('n and save_every cannot both be None.')

        if save_every is None:
            save_every = n + 1

        filename = f'{self.H}_{self.L}_{self.c}'
        if prefix is not None:
            filename = f'{prefix}_{filename}'
        if folder is not None:
            filename = f'{folder}/{filename}'

        for i in tqdm(count(1), total=n, desc='Number of realizations', ncols=120):
            while not self._make_one_realization(device=device):
                pass

            if i % save_every == 0:
                self.save(filename=filename)

            if i == n:
                break

        if i % save_every != 0:
            self.save(filename=filename)

    def get_mean_sampling(self, name: str, n_dist: int = 1000):
        x = np.array(self.__dict__[name]).reshape(self.n_realizations, -1)
        inds = np.random.randint(low=0, high=self.n_realizations, size=(self.n_realizations, n_dist))
        y = x[inds, :].mean(axis=0)
        if x.shape[1] == 1:
            y = y[..., 0]

        # r = []
        #
        # for i in range(x.shape[0]):
        #     r.append(np.random.choice(x[:, 0], size=(self.n_realizations, n_dist), replace=True).mean(axis=-1))

        return y