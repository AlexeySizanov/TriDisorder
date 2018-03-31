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
        self.conn[  xs[xs_pi] + ys[xs_pi] * self.L, xs_p[xs_pi] + ys[xs_pi] * self.L] = True
        self.conn[xs_p[xs_pi] + ys[xs_pi] * self.L,   xs[xs_pi] + ys[xs_pi] * self.L] = True

        xs_m = xs - 1
        xs_mi = (xs_m >= 0) & np.isin(xs_m + ys * self.L, inds)
        self.conn[  xs[xs_mi] + ys[xs_mi]  * self.L, xs_m[xs_mi] + ys[xs_mi] * self.L] = True
        self.conn[xs_m[xs_mi] + ys[xs_mi]  * self.L,   xs[xs_mi] + ys[xs_mi] * self.L] = True


        ys_p = ys + 1
        ys_pi = (ys_p < self.L) & np.isin(xs + ys_p * self.L, inds)
        self.conn[xs[ys_pi] +   ys[ys_pi] * self.L, xs[ys_pi] + ys_p[ys_pi] * self.L] = True
        self.conn[xs[ys_pi] + ys_p[ys_pi] * self.L, xs[ys_pi] +   ys[ys_pi] * self.L] = True

        ys_m = ys - 1
        ys_mi = (ys_m >= 0) & np.isin(xs + ys_m * self.L, inds)
        self.conn[xs[ys_mi] +   ys[ys_mi] * self.L, xs[ys_mi] + ys_m[ys_mi] * self.L] = True
        self.conn[xs[ys_mi] + ys_m[ys_mi] * self.L, xs[ys_mi] +   ys[ys_mi] * self.L] = True


        ipm = (xs_p < self.L) & (ys_m >= 0) & np.isin(xs_p + ys_m * self.L, inds)
        self.conn[  xs[ipm] +   ys[ipm] * self.L, xs_p[ipm] + ys_m[ipm] * self.L] = True
        self.conn[xs_p[ipm] + ys_m[ipm] * self.L,   xs[ipm] +   ys[ipm] * self.L] = True

        ipm = (ys_p < self.L) & (xs_m >= 0) & np.isin(xs_m + ys_p * self.L, inds)
        self.conn[  xs[ipm] +   ys[ipm] * self.L, xs_p[ipm] + ys_m[ipm] * self.L] = True
        self.conn[xs_p[ipm] + ys_m[ipm] * self.L,   xs[ipm] +   ys[ipm] * self.L] = True

        self.conn = sparse.csr_matrix(self.conn)

        self.active = np.array(self.conn.sum(axis=1)).reshape(-1) > 0
        self.inds = np.where(self.active)[0]


    def update(self, n=1):
        for _ in range(n):
            self.spins[self.inds] = self.conn.dot(self.spins)[self.inds]
            self.normalize()



    def normalize(self):
        self.spins /= np.linalg.norm(self.spins, axis=-1, keepdims=True)
