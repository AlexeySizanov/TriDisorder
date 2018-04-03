import numpy as np
from scipy import sparse


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

    def make_connections(self):
        self.conn = sparse.lil_matrix((self.N, self.N))

        inds = np.random.choice(np.arange(self.N), int(self.N * (1. - self.c)), replace=False)

        xs = inds % self.L
        ys = inds // self.L

        xs_p = xs + 1
        xs_pi = (xs_p < self.L) & np.isin(xs_p + ys * self.L, inds)
        self.conn[  xs[xs_pi] + ys[xs_pi] * self.L, xs_p[xs_pi] + ys[xs_pi] * self.L] = 1.
        self.conn[xs_p[xs_pi] + ys[xs_pi] * self.L,   xs[xs_pi] + ys[xs_pi] * self.L] = 1.

        xs_m = xs - 1
        xs_mi = (xs_m >= 0) & np.isin(xs_m + ys * self.L, inds)
        self.conn[  xs[xs_mi] + ys[xs_mi]  * self.L, xs_m[xs_mi] + ys[xs_mi] * self.L] = 1.
        self.conn[xs_m[xs_mi] + ys[xs_mi]  * self.L,   xs[xs_mi] + ys[xs_mi] * self.L] = 1.


        ys_p = ys + 1
        ys_pi = (ys_p < self.L) & np.isin(xs + ys_p * self.L, inds)
        self.conn[xs[ys_pi] +   ys[ys_pi] * self.L, xs[ys_pi] + ys_p[ys_pi] * self.L] = 1.
        self.conn[xs[ys_pi] + ys_p[ys_pi] * self.L, xs[ys_pi] +   ys[ys_pi] * self.L] = 1.

        ys_m = ys - 1
        ys_mi = (ys_m >= 0) & np.isin(xs + ys_m * self.L, inds)
        self.conn[xs[ys_mi] +   ys[ys_mi] * self.L, xs[ys_mi] + ys_m[ys_mi] * self.L] = 1.
        self.conn[xs[ys_mi] + ys_m[ys_mi] * self.L, xs[ys_mi] +   ys[ys_mi] * self.L] = 1.


        ipm = (xs_p < self.L) & (ys_m >= 0) & np.isin(xs_p + ys_m * self.L, inds)
        self.conn[  xs[ipm] +   ys[ipm] * self.L, xs_p[ipm] + ys_m[ipm] * self.L] = 1.
        self.conn[xs_p[ipm] + ys_m[ipm] * self.L,   xs[ipm] +   ys[ipm] * self.L] = 1.

        ipm = (ys_p < self.L) & (xs_m >= 0) & np.isin(xs_m + ys_p * self.L, inds)
        self.conn[  xs[ipm] +   ys[ipm] * self.L, xs_p[ipm] + ys_m[ipm] * self.L] = 1.
        self.conn[xs_p[ipm] + ys_m[ipm] * self.L,   xs[ipm] +   ys[ipm] * self.L] = 1.

        self.singles = np.array(self.conn.sum(axis=0)).reshape(-1) == 0
        self.singles_inds = np.where(self.singles)[0]

        self.conn_singles = self.conn.copy()
        self.conn_singles[self.singles_inds, self.singles_inds] = -1.

        self.conn = sparse.csr_matrix(self.conn)
        self.conn_singles = sparse.csr_matrix(self.conn_singles)


        self.active = np.array(self.conn.sum(axis=1)).reshape(-1) > 0
        self.active_inds = np.where(self.active)[0]

    def new_state(self, field=None):
        if field is None:
            new_spins = - self.conn_singles.dot(self.spins)

        else:
            new_spins = - self.conn.dot(self.spins) - field

        new_spins /= np.linalg.norm(new_spins, axis=1, keepdims=True)
        return new_spins

    def optimize(self, field=None, threshold=1e-10):
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

    def measure(self):
        pass #TODO: measure method

    def plot(self, filename=None):
        pass  #TODO: plot method


    def normalize(self):
        self.spins /= np.linalg.norm(self.spins, axis=-1, keepdims=True)
