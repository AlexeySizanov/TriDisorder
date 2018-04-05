import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt


zero_field = np.array([0., 0., 0.])

class TDSystem:
    def __init__(self, L, c):
        self.L = L
        self.N = L ** 2
        self.c = c

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

        ipm = (ys_p < self.L) & (xs_m >= 0) & np.isin(xs_m + ys_p * self.L, self.inds)
        self.conn[  xs[ipm] +   ys[ipm] * self.L, xs_p[ipm] + ys_m[ipm] * self.L] = 1.
        self.conn[xs_p[ipm] + ys_m[ipm] * self.L,   xs[ipm] +   ys[ipm] * self.L] = 1.

        self.ham = self.conn.copy()
        self.ham = sparse.csr_matrix(self.ham)

        self.conn[self.hole_inds, self.hole_inds] = -1.
        self.conn = sparse.csr_matrix(self.conn)

    def new_state(self, field=zero_field):
        new_spins = - self.conn.dot(self.spins) - field
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

    def _opt_step(self, fielf=zero_field):
        self.spins = self.new_state(fielf)
        print('energy =', self.energy())

    def measure(self):
        pass #TODO: measure method

    def energy(self):
        return np.einsum('ai,ai->', self.spins, self.ham.dot(self.spins)) / 2.

    def normalize(self):
        self.spins /= np.linalg.norm(self.spins, axis=-1, keepdims=True)

    def plot_lattice(self, shift=True):
        xs = self.inds % self.L
        ys = self.inds // self.L

        if shift:
            xs = xs + (ys * 0.5)

        plt.scatter(xs, ys)

        for i1 in range(self.inds.shape[0]):
            for i2 in range(i1, self.inds.shape[0]):
                ii1 = self.inds[i1]
                ii2 = self.inds[i2]
                if self.conn[ii1, ii2]:
                     plt.plot(xs[[i1, i2]], ys[[i1, i2]])

        plt.show()


