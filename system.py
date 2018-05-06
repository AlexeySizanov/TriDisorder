import numpy as np
from scipy import sparse
from scipy import fftpack
from multiprocessing import Pool

import matplotlib.pyplot as plt
import seaborn as sns

zero_field = np.array([0., 0., 0.])

class TDSystem:
    def __init__(self, L, c, periodic=False):
        self.L = L
        self.N = L ** 2
        self.c = c
        self.periodic=periodic

        self.make_spins()
        self.make_connections()

    def make_spins(self):
        cosThetas = np.random.rand(self.N) * 2 - 1
        Rxys = np.sqrt(1 - cosThetas**2)
        phis = np.random.rand(self.N) * 2 * np.pi

        self.spins = np.array([
            Rxys * np.cos(phis),
            Rxys * np.sin(phis),
            cosThetas
        ]).transpose((1, 0))

        all_inds = np.arange(self.N)
        self.inds = np.random.choice(all_inds, int(self.N * (1. - self.c)), replace=False)
        self.inds.sort()
        self.hole_inds = all_inds[~np.isin(all_inds, self.inds)]

        self.n_spins = self.inds.shape[0]
        # self.spins[self.hole_inds] *= 0.

    def make_connections(self):
        self.conn = sparse.lil_matrix((self.N, self.N))

        xs = self.inds % self.L
        ys = self.inds // self.L

        xs_p = xs + 1
        xs_pi = (xs_p < self.L) & np.isin(xs_p + ys * self.L, self.inds)
        self.conn[  xs[xs_pi] + ys[xs_pi] * self.L, xs_p[xs_pi] + ys[xs_pi] * self.L] = 1.
        self.conn[xs_p[xs_pi] + ys[xs_pi] * self.L,   xs[xs_pi] + ys[xs_pi] * self.L] = 1.

        xs_m = xs - 1
        xs_mi = (xs_m >= 0) & np.isin(xs_m + ys * self.L, self.inds)
        self.conn[  xs[xs_mi] + ys[xs_mi]  * self.L, xs_m[xs_mi] + ys[xs_mi] * self.L] = 1.
        self.conn[xs_m[xs_mi] + ys[xs_mi]  * self.L,   xs[xs_mi] + ys[xs_mi] * self.L] = 1.


        ys_p = ys + 1
        ys_pi = (ys_p < self.L) & np.isin(xs + ys_p * self.L, self.inds)
        self.conn[xs[ys_pi] +   ys[ys_pi] * self.L, xs[ys_pi] + ys_p[ys_pi] * self.L] = 1.
        self.conn[xs[ys_pi] + ys_p[ys_pi] * self.L, xs[ys_pi] +   ys[ys_pi] * self.L] = 1.

        ys_m = ys - 1
        ys_mi = (ys_m >= 0) & np.isin(xs + ys_m * self.L, self.inds)
        self.conn[xs[ys_mi] +   ys[ys_mi] * self.L, xs[ys_mi] + ys_m[ys_mi] * self.L] = 1.
        self.conn[xs[ys_mi] + ys_m[ys_mi] * self.L, xs[ys_mi] +   ys[ys_mi] * self.L] = 1.

        ipm = (xs_p < self.L) & (ys_m >= 0) & np.isin(xs_p + ys_m * self.L, self.inds)
        self.conn[  xs[ipm] +   ys[ipm] * self.L, xs_p[ipm] + ys_m[ipm] * self.L] = 1.
        self.conn[xs_p[ipm] + ys_m[ipm] * self.L,   xs[ipm] +   ys[ipm] * self.L] = 1.

        # ipm = (xs_p < self.L) & (ys_m >= 0) & np.isin(xs_p + ys_m * self.L, self.inds)
        # self.conn[  xs[ipm] +   ys[ipm] * self.L, xs_p[ipm] + ys_m[ipm] * self.L] = 1.
        # self.conn[xs_p[ipm] + ys_m[ipm] * self.L,   xs[ipm] +   ys[ipm] * self.L] = 1.
        #
        # ipm = (ys_p < self.L) & (xs_m >= 0) & np.isin(xs_m + ys_p * self.L, self.inds)
        # self.conn[  xs[ipm] +   ys[ipm] * self.L, xs_p[ipm] + ys_m[ipm] * self.L] = 1.
        # self.conn[xs_p[ipm] + ys_m[ipm] * self.L,   xs[ipm] +   ys[ipm] * self.L] = 1.

        if self.periodic:
            i_x_start = np.arange(self.L)
            i_x_end = np.arange(self.L) + self.L * (self.L - 1)
            self.conn[i_x_start, i_x_end] = 1.
            self.conn[i_x_end, i_x_start] = 1.


            i_y_start = np.arange(self.L) * self.L
            i_y_end = np.concatenate([
                np.arange(self.L // 2, self.L), np.arange(0, self.L // 2)
            ]) * self.L + (self.L - 1)
            self.conn[i_y_start, i_y_end] = 1.
            self.conn[i_y_end, i_y_start] = 1.

            self.conn[self.hole_inds, :] = 0.
            self.conn[:, self.hole_inds] = 0.


        self.ham = self.conn.copy()
        self.ham = sparse.csr_matrix(self.ham)

        self.conn[self.hole_inds, self.hole_inds] = -1.
        self.conn = sparse.csr_matrix(self.conn)

    def molecular_field(self, field=zero_field):
        return - self.conn.dot(self.spins) - field

    def new_state(self, field=zero_field):
        new_spins = self.molecular_field(field)
        return new_spins / np.linalg.norm(new_spins, axis=1, keepdims=True)

    def optimize(self, field=zero_field, threshold=1e-10):
        n = 0
        while True:
            new_spins = self.new_state(field=field)
            n += 1
            if n % 1000 == 0:
                diffs = np.linalg.norm(new_spins - self.spins, axis=1)
                print('n = {},   diff = {}'.format(n, diffs.mean()))
                if diffs.max() <= threshold:
                    self.spins = new_spins
                    return n
            self.spins = new_spins


    def adam(self, lr=0.1, n_steps=1, frac=1.0, field=zero_field, beta1=0.9, beta2=0.9):
        v = np.zeros_like(self.spins)
        G = np.zeros_like(self.spins)
        n = int(self.n_spins * frac)
        for t in range(n_steps):
            ds = self.molecular_field(field)

            ds = ds - np.einsum('ai,aj,aj->ai', self.spins, self.spins, ds)
            # v = v - np.einsum('ai,aj,aj->ai', self.spins, self.spins, v)

            v = (1. - beta1) * ds + beta1 * v
            G = (1. - beta2) * (ds**2) + beta2 * G

            vt = v / (1. - beta1**(t+1))
            Gt = G / (1. - beta2**(t+1))

            inds = np.random.choice(self.inds, n, replace=False)
            tmp = np.sqrt(Gt[inds] + 1e-10)

            self.spins[inds] += lr * vt[inds] / tmp
            self.normalize()

    def opt_step(self, field=zero_field, frac=0.5, out=True):
        n = int(self.n_spins * frac)
        inds = np.random.choice(self.inds, n, replace=False)
        self.spins[inds] = self.new_state(field=field)[inds]
        if out:
            print('energy =', self.measure_energy_density())

    def opt_steps(self, n_steps: int=1, frac=1.0, field: np.ndarray=zero_field):
        for _ in range(n_steps-1):
            self.opt_step(field=field, out=False, frac=frac)
        # self.opt_step(field=field, out=True, frac=frac)


    def sgd_step(self, field=zero_field, lr=0.1, frac=1.0):
        mf = self.molecular_field(field)
        ds1 = mf - self.spins
        ds = ds1 - np.einsum('ai,aj,aj->ai', self.spins, self.spins, ds1)

        n = int(self.n_spins * frac)
        inds = np.random.choice(self.inds, n, replace=False)

        self.spins[inds] += ds[inds] * lr
        self.normalize()

    def sgd(self, field=zero_field, lr=0.1, n_steps=10, frac=0.5, out=True):
        for _ in range(n_steps):
            self.sgd_step(field=field, lr=lr, frac=frac)
        # if out:
        #     print(self.energy_density())

    def measure_all(self):
        self.measure_fourier()
        self.measure_energy_density()

    def measure_fourier(self):
        self.spins[self.hole_inds] = np.array([0., 0., 0.])
        self.fourier = (np.abs(fftpack.fft2(self.spins.reshape(self.L, self.L, 3), axes=(0, 1)))**2).sum(axis=-1)
        self.spins[self.hole_inds] = np.array([1., 0., 0.])

    def make_random(self):
        phis = np.random.rand(self.N) * 2 * np.pi
        cos_thetas = np.random.rand(self.N) * 2 - 1
        sin_thetas = np.sqrt(1 - cos_thetas**2)
        self.spins = np.array([sin_thetas * np.cos(phis), sin_thetas * np.sin(phis), cos_thetas]).transpose(1, 0    )

    def randomize(self, rate=0.1):
        phis = np.random.rand(self.N) * 2 * np.pi
        cos_thetas = np.random.rand(self.N) * 2 - 1
        sin_thetas = np.sqrt(1 - cos_thetas**2)
        self.spins += rate * np.array([sin_thetas * np.cos(phis), sin_thetas * np.sin(phis), cos_thetas]).transpose(1, 0    )
        self.normalize()

    def make_helix_xy(self):
        xs = np.arange(self.L).reshape(-1, 1)
        ys = np.arange(self.L).reshape(1, -1)
        phis = np.pi * (4 * xs + 2 * ys) / 3
        spins = np.array([np.cos(phis), np.sin(phis), np.zeros_like(phis)]).transpose(1, 2, 0)
        self.spins = spins.reshape(-1, 3)


    def measure_energy_density(self):
        self.energy_density = np.einsum('ai,ai->', self.spins, self.ham.dot(self.spins)) / (2. * self.n_spins)

    def normalize(self):
        self.spins /= np.linalg.norm(self.spins, axis=-1, keepdims=True)

    def make_xy(self):
        self.spins[:, 2] = 0.
        self.normalize()

    def rot_xy(self, alpha):
        alpha =alpha * np.pi / 180.
        R = np.array([
            [np.cos(alpha), np.sin(alpha), 0],
            [-np.sin(alpha), np.cos(alpha), 0],
            [0., 0., 1.]
        ])
        self.spins = np.einsum('ij,aj->ai', R, self.spins)

    def plot(self, spins=True, lattice=True, shift=True):
        xs = self.inds % self.L
        ys = self.inds // self.L

        if shift:
            xs = xs + (ys * 0.5)

        plt.scatter(xs, ys)

        if lattice:
            for i1 in range(self.inds.shape[0]):
                for i2 in range(i1, self.inds.shape[0]):
                    ii1 = self.inds[i1]
                    ii2 = self.inds[i2]
                    if self.conn[ii1, ii2]:
                         plt.plot(xs[[i1, i2]], ys[[i1, i2]])

        if spins:
            for i in range(self.inds.shape[0]):
                plt.plot([xs[i], xs[i] + self.spins[self.inds[i], 0]/2], [ys[i], ys[i] + self.spins[self.inds[i], 1]/2], color='black')

    def plot_fourier(self):
        plt.imshow(self.fourier)



class TDResult:
    @classmethod
    def empty(cls):
        return TDResult(None, None, None, None, 0)

    def __init__(self, L, c, energy_density, fourier, n):
        self.n = n

        self.L = L
        self.c = c
        self.energy_density = energy_density
        self.fourier = fourier


    def copy(self):
        res = TDResult.empty()
        res.L = self.L
        res.c = self.c
        res.energy_density = self.energy_density
        res.fourier = self.fourier.copy()
        return res

    def _add__(self, other):
        """
        :type other: TDResult
        :rtype: TDResult
        """
        if (self.L != other.L) or (self.c != other.c):
            raise Exception('TDResult.__add__ : systems must be similar.')

        if self.n == 0:
            if other.n == 0:
                raise Exception('TDResult.__add__ : both results are empty')
            else:
                return other.copy()
        elif other.n == 0:
            return self.copy()


        res = TDResult.empty()

        res.L = self.L
        res.c = self.c
        res.n = self.n + other.n

        res.energy_density = (self.energy_density * self.n + other.energy_density * other.n) / res.n
        res.fourier = (self.fourier * self.n + other.fourier * other.n) / res.n

        return res

