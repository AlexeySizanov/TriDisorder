import numpy as np
from scipy import sparse
from scipy import fftpack
from multiprocessing import Pool

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange

import torch

zero_field = torch.tensor([0., 0., 0.])
ez = torch.tensor([0., 0., 1.])

class TDSystem:
    def __init__(self, L, c, periodic=False, cuda=True):
        self.L = L
        self.N = L ** 2
        self.Z = 6  # neighbors number in regular lattice
        self.c = c
        self.periodic=periodic
        self.cuda = True

        self.measures = dict()

        self.make_spins()
        self.make_connections()

    def make_spins(self):
        thetas = np.arccos(np.random.rand(self.N + 1) * 2 - 1).astype(np.float32)

        self.thetas = torch.from_numpy(thetas)
        self.phis = torch.rand(self.N) * 2 * np.pi

        if self.cuda:
            self.thetas = self.thetas.cuda()
            self.phis = self.phis.cuda()

        self.thetas.requires_grad = True
        self.phis.requires_grad = True

        hole_inds = np.random.choice(np.arange(self.N), int(self.N * self.c), replace=False)
        self.hole_inds = np.hstack([self.N, hole_inds])

        self.hole_inds = torch.from_numpy(self.hole_inds)


    def spins(self):
        sinTh = torch.sin(self.thetas)

        xs = sinTh * torch.cos(self.phis)
        ys = sinTh * torch.sin(self.phis)
        zs = torch.cos(self.thetas)

        xs.data[self.hole_inds] = 0.
        ys.data[self.hole_inds] = 0.
        zs.data[self.hole_inds] = 0.

        return xs, ys, zs

    def spins_numpy(self):
        sinTh = np.sin(self.thetas.data.cpu().numpy())

        xs = sinTh * np.cos(self.phis.data.cpu().numpy())
        ys = sinTh * np.sin(self.phis.data.cpu().numpy())
        zs = np.cos(self.thetas.data.cpu().numpy())

        xs[self.hole_inds.numpy()] = 0.
        ys[self.hole_inds.numpy()] = 0.
        zs[self.hole_inds.numpy()] = 0.

        return np.array([xs, ys, zs]).T


    def inds_from_xy(self, x, y):
        return y * self.L + x
    def xy_from_inds(self, inds):
        return inds % self.L, inds // self.L

    def make_connections(self):
        xs, ys = self.xy_from_inds(np.arange(self.N + 1))

        conn = np.array([
            self.inds_from_xy(xs + 1, ys),
            self.inds_from_xy(xs - 1, ys),
            self.inds_from_xy(xs, ys + 1),
            self.inds_from_xy(xs, ys - 1),
            self.inds_from_xy(xs + 1, ys - 1),
            self.inds_from_xy(xs - 1, ys + 1)
        ]).T

        conn[xs == 0, [[1], [5]]] = self.N
        conn[xs == self.L - 1, [[0], [4]]] = self.N
        conn[ys == 0, [[3], [5]]] = self.N
        conn[ys == self.L - 1, [[2], [6]]] = self.N
        conn[self.N, :] = self.N

        self.conn = torch.from_numpy(conn)

    def molecular_field_numpy(self):
        return self.spins_numpy()[self.conn.numpy()].sum(axis=1)

    def energy_density(self, field=zero_field):
        spins = self.spins()
        mf = [v[self.conn].sum(dim=1) for v in spins]
        res = sum([v * vmf for v, vmf in zip(spins, mf)]).sum() / (2 * self.N)
        res.add_(sum(-v * fv for v, fv in zip(spins, field)).mean())
        self.measures['energy density'] = float(res)
        return res


    def measure_energy_density_field(self, field=zero_field):
        spins = self.spins_numpy()
        mf = self.molecular_field_numpy()
        self.measures['energy density field'] =  np.einsum('ai,ai->a', spins, mf / 2 - field)

    def measure_all(self):
        self.measure_fourier()
        self.measure_energy_density_field()
        self.measure_plane()

    def measure_fourier(self):
        spins = self.spins_numpy()
        self.measures['fourier'] = (np.abs(fftpack.fft2(spins.reshape(self.L, self.L, 3), axes=(0, 1)))**2).sum(axis=-1)


    def make_helix_xy(self):
        xs = np.arange(self.L).reshape(-1, 1)
        ys = np.arange(self.L).reshape(1, -1)
        phis = np.pi * (4 * xs + 2 * ys) / 3
        self.phis.data.copy_(torch.tensor(phis, dtype=torch.float32))
        self.thetas.data *= 0

    def make_xy(self):
        self.thetas.data[:] = np.pi / 2

    def plot(self, spins=True, lattice=True, shift=True):
        xs = self.inds % self.L
        ys = self.inds // self.L

        if shift:
            xs = xs + (ys * 0.5)

        plt.scatter(xs, ys, marker='.')

        if lattice:
            for i1 in trange(self.inds.shape[0]):
                for i2 in range(i1, self.inds.shape[0]):
                    ii1 = self.inds[i1]
                    ii2 = self.inds[i2]
                    if self.conn[ii1, ii2]:
                         plt.plot(xs[[i1, i2]], ys[[i1, i2]])

        if spins:
            for i in trange(self.inds.shape[0]):
                plt.plot([xs[i], xs[i] + self.spins[self.inds[i], 0]/2], [ys[i], ys[i] + self.spins[self.inds[i], 1]/2], color='black')

    def plot_fourier(self):
        plt.imshow(self.fourier)

    def plot_density(self, hm=False):
        if hm:
            sns.heatmap(self.density.reshape(self.L, self.L))
        else:
            plt.imshow(self.density.reshape(self.L, self.L))

    def measure_plane(self):
        xs, ys = self.xy_from_inds(np.arange(self.N))

        spins = self.spins_numpy()[:-1]

        xs_pi = (xs + 1 < self.L)
        ix_pi = (xs + 1)[xs_pi] + ys[xs_pi] * self.L
        
        xs_mi = (xs - 1 < self.L)
        ix_mi = (xs - 1)[xs_mi] + ys[xs_mi] * self.L

        ys_pi = (ys + 1 < self.L)
        iy_pi = xs[ys_pi] + (ys + 1)[ys_pi] * self.L

        ys_mi = (ys - 1 < self.L)
        iy_mi = xs[ys_mi] + (ys - 1)[ys_mi] * self.L

        res = np.zeros_like(self.spins)

        #TODO: diagonal interactions!
        res[xs_pi] += np.cross(self.spins[xs_pi], self.spins[ix_pi])
        res[xs_mi] -= np.cross(self.spins[xs_mi], self.spins[ix_mi])
        res[ys_pi] += np.cross(self.spins[ys_pi], self.spins[iy_pi])
        res[ys_mi] -= np.cross(self.spins[ys_mi], self.spins[iy_mi])

        self.plane = res / self.n_neighbors.reshape(-1, 1)

    def plot_plane(self, i=2, hm=True):
        values = self.plane[:, i].reshape(self.L, self.L)
        if hm:
            sns.heatmap(values)
        else:
            plt.imshow(values)
