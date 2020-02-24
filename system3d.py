from typing import Tuple, Optional

import numpy as np
from scipy import fftpack, sparse
from multiprocessing import Pool

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange

import torch, torch.sparse


class TDSystem3DXY:
    """
    Variables:   XY spins.
    Lattice:     Stacked rhombic layers. Each rhombic layer (xy-plane) has triangular lattice.
    Interaction: NN(xy, J_xy) + NN(z, J_z),
    Disorder:    Non-magnetic impurities.
    Boundaries:  Open BC.
    """
    def __init__(self, L: int, H: int, c: float, field=None, device: str = 'cuda'):
        """
        :param L: Side of the rhombus.
        :param H: Height of the stack.
        :param c: impurities concentration.
        :param field: 2d (xy) vector of the field.
        :param device: device for computations.
        """
        if device not in ('cuda', 'cpu'):
            raise ValueError('`device` must be "cuda" (default) or "cpu".')

        self.L = L
        self.H = H
        self.Nxy = L ** 2
        self.N = H * self.Nxy
        self.c = c
        self._device = device

        self.measures = dict()

        self.make_connections()
        self.make_spins()
        self.field = np.zeros(3) if field is None else field

    @classmethod
    def empty(cls, device='cuda'):
        return TDSystem3DXY(L=1, H=1, c=0., device=device)

    @classmethod
    def load(cls, filename, device='cuda'):
        s = TDSystem3DXY.empty(device=device)
        s._load(filename)

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, field):
        self._field = torch.tensor(field, dtype=torch.float32, device=self._device)



    def inds_from_xyz(self, x, y, z):
        return x + y * self.L + z * self.Nxy

    def xyz_from_inds(self, inds):
        z = inds // self.Nxy
        xy = inds % self.Nxy
        return xy % self.L, xy // self.L, z

    def make_connections(self):
        all_inds = np.arange(self.N)  # indices of all nodes (spins + holes)
        hole_inds = np.random.choice(all_inds, int(self.N * self.c), replace=False)  # indices of the holes
        hole_inds = np.hstack([hole_inds, self.N])
        spin_inds = np.setdiff1d(all_inds, hole_inds)  # indices of the spins

        xs, ys, zs = self.xyz_from_inds(all_inds)  # x, y, z coords of all nodes.

        # array of the bonds i.e. indices array of the shape (N, 8), 8 corresponds to 8 bonds in the pure system.

        all_bonds = np.array([
                self.inds_from_xyz(xs + 1, ys, zs),  # 0
                self.inds_from_xyz(xs - 1, ys, zs),  # 1
                self.inds_from_xyz(xs, ys + 1, zs),  # 2
                self.inds_from_xyz(xs, ys - 1, zs),  # 3
                self.inds_from_xyz(xs + 1, ys - 1, zs),  # 4
                self.inds_from_xyz(xs - 1, ys + 1, zs),  # 5
                self.inds_from_xyz(xs, ys, zs + 1),  # 6
                self.inds_from_xyz(xs, ys, zs - 1)   # 7
        ], order='F').T  # shape ~=(N, 8)
        all_bonds[xs == 0, [[1], [5]]] = self.N
        all_bonds[xs == self.L - 1, [[0], [4]]] = self.N
        all_bonds[ys == 0, [[3], [4]]] = self.N
        all_bonds[ys == self.L - 1, [[2], [5]]] = self.N
        all_bonds[zs == 0, [[6]]] = self.N
        all_bonds[zs == self.H - 1, [[7]]] = self.N
        all_bonds = np.vstack([all_bonds, [self.N] * 8])  # shape = (N + 1, 8)
        all_bonds[hole_inds, :] = self.N
        all_bonds[np.isin(all_bonds, hole_inds)] = self.N

        self.bonds_np = all_bonds  # shape = (N + 1, 8)
        self.bonds = torch.tensor(all_bonds, device=self._device)

        conn = sparse.lil_matrix((self.N, self.N))
        real_bonds = np.vstack([np.repeat(all_inds, 8), all_bonds[:-1].reshape(-1)])  # shape = (2, N*8)
        real_bonds = real_bonds[:, real_bonds[1] != self.N]  # shape = (2, N_real_bonds)
        conn[real_bonds[0], real_bonds[1]] = 1

        self.conn_np = conn.tocsr()
        self.conn = torch.sparse.FloatTensor(torch.LongTensor(real_bonds),
                                             torch.ones(real_bonds.shape[1]),
                                             torch.Size([self.N, self.N]))

        self.all_inds = torch.from_numpy(all_inds)
        self.hole_inds = torch.from_numpy(hole_inds)
        self.spin_inds = torch.from_numpy(spin_inds)


    def make_spins(self):
        self.thetas = torch.rand(self.N + 1, dtype=torch.float32, device=self._device, requires_grad=True) * 2 * np.pi

    def randomize(self):
        self.thetas[:] = torch.rand(self.N + 1, dtype=torch.float32) * 2 * np.pi

    def spins(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: Tuple[torch.Tensor[shape=(N+1,)], torch.Tensor[shape=(N+1,)]]
        """
        sx = torch.cos(self.thetas)
        sy = torch.sin(self.thetas)

        sx[self.hole_inds] = 0.
        sy[self.hole_inds] = 0.

        return sx, sy

    def spins_numpy(self) -> np.array:
        """
        shape = (2, N)
        """
        sx, sy = self.spins()
        sx = sx.data.cpu().numpy()[:-1]
        sy = sy.data.cpu().numpy()[:-1]
        return np.array([sx, sy])

    def energy_density(self):
        sx, sy = self.spins()
        mx, my = sx[self.bonds].sum(dim=1) + self._field[0], sy[self.bonds].sum(dim=1) + self._field[1]
        res = (sx * mx + sy * my) / (2 * self.N)
        self.measures['energy density'] = float(res)
        return res

    def measure_all(self):
        self.measure_fourier()
        self.measure_energy_density_field()

    def molecular_field_numpy(self):
        return self.spins_numpy()[self.bonds_np].sum(axis=1)[:-1]

    def measure_energy_density_field(self):
        field = self.field.cpu().numpy()
        spins = self.spins_numpy()[:-1]
        mf = self.molecular_field_numpy()
        self.measures['energy density field'] =  np.einsum('ai,ai->a', spins, mf / 2 - field)

    def measure_fourier(self):
        spins = self.spins_numpy()[:-1]
        self.measures['fourier'] = (np.abs(fftpack.fft2(spins.reshape(self.L, self.L, 3), axes=(0, 1)))**2).sum(axis=-1)

    def plot(self, z: Optional[int] = None, draw_spins: bool = True,
             draw_lattice: bool = False, distortion: bool = False):

        plt.figaspect(1 + 1/np.sqrt(2))
        z0 = self.H // 2 if z is None else z

        spin_inds = self.spin_inds.numpy()
        xs, ys, zs = [component[spin_inds] for component in self.xyz_from_inds(np.arange(self.N))]
        z_mask = zs == z0
        spin_inds = spin_inds[z_mask]
        xs, ys = xs[z_mask], ys[z_mask]

        spins = self.spins_numpy()[:, spin_inds]

        xs = xs + (ys * 0.5)
        ys = ys * np.sqrt(2) / 2

        if distortion:
            xs = xs.astype(float) + np.random.rand(xs.shape[0]) * 0.2
            ys = ys.astype(float) + np.random.rand(xs.shape[0]) * 0.2

        plt.scatter(xs, ys, marker='.')

        if draw_lattice:
            for i1 in trange(len(xs)):
                bond_inds = self.bonds_np[spin_inds[i1]]
                bond_inds = np.isin(spin_inds, bond_inds).nonzero()[0]
                for i2 in bond_inds:
                     plt.plot(xs[[i1, i2]], ys[[i1, i2]], color=(0,0,0,0.2))

        if draw_spins:
            # for i in range(len(spins)):
            #     plt.plot([xs[i], xs[i] + spins[i, 0]/2], [ys[i], ys[i] + spins[i, 1]/2], color=(0.2,1,0.2))
            for i in range(len(xs)):
                plt.arrow(x=xs[i] - spins[0, i] / 4, y=ys[i] - spins[1, i] / 4,
                          dx=spins[0, i] / 2, dy=spins[1, i] / 2,
                          length_includes_head=True,
                          head_width=0.2, head_length=0.5,
                          color=(0, 0.5, 0.8))


    def plot_fourier(self):
        plt.imshow(self.measures['fourier'])

    def plot_density(self, size=10):

        den = self.measures['energy density field']
        den -= den.min()
        den /= den.max()

        colors = np.zeros((den.shape[0], 3))
        colors[:,0] = den ** 0.5
        # colors[:, 0] = -np.log(0.1 + den*0.9) / np.log(0.1)

        xs, ys = self.xyz_from_inds(np.arange(self.N))
        xs = xs + ys / 2

        plt.scatter(xs[self.inds], ys[self.inds], s=size**2, c=colors[self.inds, 0])
        # plt.scatter(xs[self.inds], ys[self.inds], s=size, c=colors[self.inds], norm=plt.Normalize(0, 1))



    def save(self, filename):
        with open(filename, 'wb') as f:
            np.savez(f,
                     angles=self.thetas.data.cpu().numpy(),
                     bonds=self.bonds_np,
                     L=[self.L,],
                     H=[self.H,],
                     c=[self.c],
                     filed=self.field.cpu().numpy(),
                     )

    def _load(self, filename):
        with open(filename, 'rb') as f:
            data = np.load(f)
            self.angles = torch.tensor(data['angles'], device=self._device)
            self.bonds_np = data['bonds']
            self.bonds = torch.tensor(self.bonds_np, device=self._device)
            self.L = data['L'][0]
            self.H = data['H'][0]
            self.c = data['c'][0]
            self.field = data['field']

    def cuda(self):
        if not self._device == 'cuda':
            self._device = 'cuda'
            self.thetas = self.thetas.cuda()
            self.bonds = self.bonds.cuda()

    def cpu(self):
        if self._device == 'cpu':
            self._device = 'cpu'
            self.thetas = self.thetas.cpu()
            self.bonds = self.bonds.cpu()