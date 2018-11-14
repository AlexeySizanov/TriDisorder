import numpy as np
from scipy import fftpack
from multiprocessing import Pool

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange

import torch


class TDSystem:
    def __init__(self, L, c, field=None, cuda=True):
        self.L = L
        self.N = L ** 2
        self.c = c
        self._cuda = cuda

        self.measures = dict()

        self.make_spins()
        self.make_connections()
        self.field = np.zeros(3) if field is None else field

    @classmethod
    def empty(cls, cuda=True):
        return TDSystem(1, 0, cuda=cuda)

    @classmethod
    def load(cls, filename, cuda=True):
        s = TDSystem.empty(cuda=cuda)
        s._load(filename)

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, field):
        self._field = torch.tensor(field, dtype=torch.float32, device = 'cuda' if self._cuda else 'cpu')


    def make_spins(self):
        thetas = np.arccos(np.random.rand(self.N + 1) * 2 - 1).astype(np.float32)

        # self.thetas = torch.from_numpy(thetas)
        # self.phis = torch.rand(self.N + 1) * 2 * np.pi

        self.angles = torch.zeros(self.N+1, 2)
        self.angles[:, 0] = torch.rand(self.N + 1) * 2 * np.pi # phis
        self.angles[:, 1] = torch.from_numpy(thetas) # thetas

        if self._cuda:
            # self.thetas = self.thetas.cuda()
            # self.phis = self.phis.cuda()
            self.angles = self.angles.cuda()

        # self.thetas.requires_grad = True
        # self.phis.requires_grad = True
        self.angles.requires_grad = True

    def randomize(self, xy=False):
        thetas = np.arccos(np.random.rand(self.N + 1) * 2 - 1).astype(np.float32)

        # self.thetas = torch.from_numpy(thetas)
        # self.phis = torch.rand(self.N + 1) * 2 * np.pi
        device = 'cuda' if self._cuda else 'cpu'
        self.angles.data[:, 0] = torch.rand(self.N + 1, device=device) * 2 * np.pi # phis
        self.angles.data[:, 1] = torch.tensor(thetas, device=device) if not xy else np.pi/2 # thetas

    def spins(self):
        phis, thetas = self.angles[:, 0], self.angles[:, 1]

        sinTh = torch.sin(thetas)

        xs = sinTh * torch.cos(phis)
        ys = sinTh * torch.sin(phis)
        zs = torch.cos(thetas)

        xs.data[self.hole_inds] = 0.
        ys.data[self.hole_inds] = 0.
        zs.data[self.hole_inds] = 0.

        return xs, ys, zs

    def spins_numpy(self):
        phis, thetas = self.angles[:, 0], self.angles[:, 1]
        sinTh = np.sin(thetas.data.cpu().numpy())

        xs = sinTh * np.cos(phis.data.cpu().numpy())
        ys = sinTh * np.sin(phis.data.cpu().numpy())
        zs = np.cos(thetas.data.cpu().numpy())

        xs[self.hole_inds.numpy()] = 0.
        ys[self.hole_inds.numpy()] = 0.
        zs[self.hole_inds.numpy()] = 0.

        return np.array([xs, ys, zs]).T


    def inds_from_xy(self, x, y):
        return y * self.L + x

    def xy_from_inds(self, inds):
        return inds % self.L, inds // self.L

    def make_connections(self):
        hole_inds = np.random.choice(np.arange(self.N), int(self.N * self.c), replace=False)
        self.hole_inds = np.hstack([self.N, hole_inds])

        self.hole_inds = torch.from_numpy(self.hole_inds)
        inds = np.arange(self.N)
        self.inds = inds[~np.isin(inds, self.hole_inds)]

        xs, ys = self.xy_from_inds(self.inds)

        self.fixed_index = ((xs - self.L / 2)**2 + (ys - self.L / 2)**2).argmin()


        xs, ys = self.xy_from_inds(np.arange(self.N))

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
        conn[ys == 0, [[3], [4]]] = self.N
        conn[ys == self.L - 1, [[2], [5]]] = self.N
        conn = np.vstack([conn, [self.N]*6])

        self.conn_numpy = conn
        # self.conn = torch.from_numpy(conn)
        self.conn = torch.tensor(conn, device='cuda' if self._cuda else 'cpu')

    def molecular_field_numpy(self):
        return self.spins_numpy()[self.conn_numpy].sum(axis=1)[:-1]

    def energy_density(self):
        spins = self.spins()
        mf = [v[self.conn].sum(dim=1) for v in spins]
        res = sum([v * vmf for v, vmf in zip(spins, mf)]).sum() / (2 * self.N)
        # res.add_(sum(-v * fv for v, fv in zip(spins, self._field)).mean())
        res -= sum(v * fv for v, fv in zip(spins, self._field)).mean()
        self.measures['energy density'] = float(res)
        return res


    def measure_energy_density_field(self):
        field = self.field.cpu().numpy()
        spins = self.spins_numpy()[:-1]
        mf = self.molecular_field_numpy()
        self.measures['energy density field'] =  np.einsum('ai,ai->a', spins, mf / 2 - field)

    def measure_all(self):
        self.measure_fourier()
        self.measure_energy_density_field()
        self.measure_plane()

    def measure_fourier(self):
        spins = self.spins_numpy()[:-1]
        self.measures['fourier'] = (np.abs(fftpack.fft2(spins.reshape(self.L, self.L, 3), axes=(0, 1)))**2).sum(axis=-1)


    def make_helix_xy(self, theta=None):
        xs = np.arange(self.L).reshape(-1, 1)
        ys = np.arange(self.L).reshape(1, -1)
        phis = np.pi * (4 * xs + 2 * ys) / 3
        self.angles.data[:-1, 0] = torch.tensor(phis.flatten(), dtype=torch.float32)
        self.angles.data[:-1, 1] = np.pi / 2 if theta is None else theta

    def make_xy(self):
        self.angles.data[:, 1] = np.pi / 2

    def make_z(self, theta=np.pi/4):
        self.angles.data[:, 1] = theta

    def plot(self, draw_spins=True, draw_lattice=True, shift=True, distortion=False, draw_z=True, size=10):
        xs, ys = self.xy_from_inds(np.arange(self.N))
        spins = self.spins_numpy()

        if distortion:
            xs = xs.astype(float) + np.random.rand(xs.shape[0]) * 0.2
            ys = ys.astype(float) + np.random.rand(xs.shape[0]) * 0.2

        if shift:
            xs = xs + (ys * 0.5)

        plt.scatter(xs, ys, marker='.')

        conn = self.conn_numpy
        holes = np.isin(conn, self.hole_inds)

        if draw_lattice:
            for i1 in tqdm(self.inds):
                for i2 in conn[i1, ~holes[i1]]:
                     plt.plot(xs[[i1, i2]], ys[[i1, i2]], color=(0,0,0,0.2))

        if draw_z:
            zs = spins[:, 2].copy()
            colors = np.zeros((zs.shape[0], 4))
            colors[:, 3] = 1.0
            colors[zs > 0, 0] = zs[zs > 0]
            colors[zs <= 0, 2] = -zs[zs <= 0]

            plt.scatter(xs[self.inds], ys[self.inds], s=size**2, c=colors[self.inds])

        if draw_spins:
            for i in tqdm(self.inds):
                plt.plot([xs[i], xs[i] + spins[i, 0]/2], [ys[i], ys[i] + spins[i, 1]/2], color=(0.2,1,0.2))


    def plot_fourier(self):
        plt.imshow(self.measures['fourier'])

    def plot_density(self, size=10):

        den = self.measures['energy density field']
        den -= den.min()
        den /= den.max()

        colors = np.zeros((den.shape[0], 3))
        colors[:,0] = den ** 0.5
        # colors[:, 0] = -np.log(0.1 + den*0.9) / np.log(0.1)

        xs, ys = self.xy_from_inds(np.arange(self.N))
        xs = xs + ys / 2

        plt.scatter(xs[self.inds], ys[self.inds], s=size**2, c=colors[self.inds, 0])
        # plt.scatter(xs[self.inds], ys[self.inds], s=size, c=colors[self.inds], norm=plt.Normalize(0, 1))



    def measure_plane(self):
        pass
        # xs, ys = self.xy_from_inds(np.arange(self.N))
        #
        # spins = self.spins_numpy()[:-1]
        #
        # xs_pi = (xs + 1 < self.L)
        # ix_pi = (xs + 1)[xs_pi] + ys[xs_pi] * self.L
        #
        # xs_mi = (xs - 1 < self.L)
        # ix_mi = (xs - 1)[xs_mi] + ys[xs_mi] * self.L
        #
        # ys_pi = (ys + 1 < self.L)
        # iy_pi = xs[ys_pi] + (ys + 1)[ys_pi] * self.L
        #
        # ys_mi = (ys - 1 < self.L)
        # iy_mi = xs[ys_mi] + (ys - 1)[ys_mi] * self.L
        #
        # res = np.zeros_like(spins)
        #
        # #TODO: diagonal interactions!
        # res[xs_pi] += np.cross(self.spins[xs_pi], self.spins[ix_pi])
        # res[xs_mi] -= np.cross(self.spins[xs_mi], self.spins[ix_mi])
        # res[ys_pi] += np.cross(self.spins[ys_pi], self.spins[iy_pi])
        # res[ys_mi] -= np.cross(self.spins[ys_mi], self.spins[iy_mi])
        #
        # self.plane = res / self.n_neighbors.reshape(-1, 1)

    def plot_plane(self, i=2, hm=True):
        pass
        # values = self.plane[:, i].reshape(self.L, self.L)
        # if hm:
        #     sns.heatmap(values)
        # else:
        #     plt.imshow(values)

    def save(self, filename):
        with open(filename, 'wb') as f:
            np.savez(f,
                     angles=self.angles.data.cpu().numpy(),
                     conn=self.conn_numpy,
                     L=[self.L,],
                     c=[self.c],
                     filed=self.field.cpu().numpy(),
                     )

    def _load(self, filename):
        with open(filename, 'rb') as f:
            data = np.load(f)
            self.angles = torch.tensor(data['angles'], device='cuda' if self._cuda else 'cpu')
            self.conn_numpy = data['conn']
            self.conn = torch.tensor(self.conn_numpy, device='cuda' if self._cuda else 'cpu')
            self.L = data['L'][0]
            self.c = data['c'][0]
            self.field = data['field']

    def cuda(self):
        if not self._cuda:
            self._cuda = True
            self.angles = self.angles.cuda()
            self.conn = self.conn.cuda()


    def cpu(self):
        if self._cuda:
            self._cuda = False
            self.angles = self.angles.cpu()
            self.conn = self.conn.cpu()