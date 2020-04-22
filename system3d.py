from typing import Tuple, Optional, Union
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.sparse
from scipy import sparse
import scipy.sparse.linalg as SLA
from tqdm import trange, tqdm

from utils.ndsparse import NDSparse


class TDSystem3D:
    """
    Variables:   3D spins.
    Lattice:     Stacked rhombic layers. Each rhombic layer (xy-plane) has triangular lattice.
    Interaction: AF-NN(xy, J=1) + AFM-NN(z, J=1),
    Disorder:    Non-magnetic impurities.
    Boundaries:  Open BC.
    """
    def __init__(self, L: int, H: int, c: float, device: str = 'cuda'):
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


    def inds_from_xyz_gl(self, x, y, z):
        return x + y * self.L + z * self.Nxy

    def xyz_from_inds_gl(self, inds):
        z = inds // self.Nxy
        xy = inds % self.Nxy
        return xy % self.L, xy // self.L, z

    def make_connections(self):
        """
        Global ("gl" suffix) indices are indices inside whole lattice.
        Local ("loc" suffix) indices are sequential indices of the magnetic sites only(0, 1, ..., #magnetic sites).
        For the magnetic site there is a global <-> local bijection.
        """
        # ---------------------------------------------------------------------------------------------
        # making indices:
        all_inds_gl = np.arange(self.N)  # indices of all nodes (spins + holes)
        hole_inds_gl = np.random.choice(all_inds_gl, int(self.N * self.c), replace=False)  # indices of the holes
        spin_inds_gl = np.setdiff1d(all_inds_gl, hole_inds_gl)  # indices of the spins
        self.n_spins = len(spin_inds_gl)

        xs, ys, zs = self.xyz_from_inds_gl(spin_inds_gl)  # x, y, z coords of all nodes.

        self.all_inds_gl = torch.from_numpy(all_inds_gl)
        self.hole_inds_gl = torch.from_numpy(hole_inds_gl)
        self.spin_inds_gl = torch.from_numpy(spin_inds_gl)

        self.local_to_global = self.spin_inds_gl

        global_to_local = np.full_like(all_inds_gl, self.N)
        global_to_local[spin_inds_gl] = np.arange(self.n_spins)

        # ---------------------------------------------------------------------------------------------
        # making bonds:
        # array of the bonds i.e. indices array of the shape (N', 8), 8 corresponds to 8 bonds in the pure system.

        all_bonds_gl = np.array([
                self.inds_from_xyz_gl(xs + 1, ys, zs),  # 0
                self.inds_from_xyz_gl(xs - 1, ys, zs),  # 1
                self.inds_from_xyz_gl(xs, ys + 1, zs),  # 2
                self.inds_from_xyz_gl(xs, ys - 1, zs),  # 3
                self.inds_from_xyz_gl(xs + 1, ys - 1, zs),  # 4
                self.inds_from_xyz_gl(xs - 1, ys + 1, zs),  # 5
                self.inds_from_xyz_gl(xs, ys, zs + 1),  # 6
                self.inds_from_xyz_gl(xs, ys, zs - 1)   # 7
        ], order='F').T  # shape ~=(N, 8)

        mask_x0 = (xs == 0); mask_x1 = (xs == self.L - 1)
        all_bonds_gl[mask_x0, [[1], [5]]] = spin_inds_gl[mask_x0]
        all_bonds_gl[mask_x1, [[0], [4]]] = spin_inds_gl[mask_x1]

        mask_y0 = (ys == 0); mask_y1 = (ys == self.L - 1)
        all_bonds_gl[mask_y0, [[3], [4]]] = spin_inds_gl[mask_y0]
        all_bonds_gl[mask_y1, [[2], [5]]] = spin_inds_gl[mask_y1]

        mask_z0 = (zs == 0); mask_z1 = (zs == self.H - 1)
        all_bonds_gl[mask_z0, [[7]]] = spin_inds_gl[mask_z0]
        all_bonds_gl[mask_z1, [[6]]] = spin_inds_gl[mask_z1]

        holes_mask = np.isin(all_bonds_gl, hole_inds_gl) | (all_bonds_gl < 0) | (all_bonds_gl >= self.N)
        all_bonds_gl[holes_mask] = np.repeat(spin_inds_gl.reshape(-1, 1), 8, axis=1)[holes_mask]
        self.n_false_bonds = holes_mask.sum()

        self.bonds_np = global_to_local[all_bonds_gl]
        self.bonds = torch.tensor(self.bonds_np, device=self._device)

        self.bond_weights_np = np.ones_like(self.bonds_np, dtype=float)
        self.bond_weights_np[self.bonds_np == np.arange(self.n_spins).reshape(-1, 1)] = 0.
        self.bond_weights = torch.tensor(self.bond_weights_np, device=self._device)

        self.inds_z0 = mask_z0.nonzero()[0]
        self.inds_z1 = mask_z1.nonzero()[0]
        self.inds_z_bounds = np.hstack([self.inds_z0, self.inds_z1])

        self.xs = xs
        self.ys = ys
        self.zs = zs


    def make_spins(self, rand_scale=0.2, rand_z_bounds: bool = False):
        self.thetas = torch.full([self.n_spins], fill_value=np.pi / 2, dtype=torch.double)
        self.phis = torch.tensor( (2 * np.pi / 3) * (self.xs - self.ys) + np.pi * self.zs) % (2 * np.pi) + np.pi / 6

        r1 = (2 * torch.rand(self.n_spins) - 1) * rand_scale
        r2 = (2 * torch.rand(self.n_spins) - 1) * rand_scale

        if not rand_z_bounds:
            r1[self.inds_z_bounds] = 0.
            r2[self.inds_z_bounds] = 0.

        self.thetas += r1
        self.phis += r2

        self.thetas = self.thetas.to(device=self._device)
        self.thetas.requires_grad = True

        self.phis = self.phis.to(self._device)
        self.phis.requires_grad = True


    def cuda(self):
        if not self._device == 'cuda':
            self._device = 'cuda'
            self.thetas = self.thetas.cuda()
            self.phis = self.phis.cuda()
            self.bonds_gl = self.bonds_gl.cuda()

    def cpu(self):
        if self._device == 'cpu':
            self._device = 'cpu'
            self.thetas = self.thetas.cpu()
            self.phis = self.phis.cpu()
            self.bonds_gl = self.bonds_gl.cpu()


    def spins(self, single: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        :return: Tuple[torch.Tensor[shape=(N+1,)], torch.Tensor[shape=(N+1,)]]
        """

        cos_theta = torch.cos(self.thetas)
        sin_theta = torch.sin(self.thetas)
        cos_phi = torch.cos(self.phis)
        sin_phi = torch.sin(self.phis)

        sx = sin_theta * cos_phi
        sy = sin_theta * sin_phi
        sz = cos_theta

        if single:
            return torch.stack([sx, sy, sz], dim=0)
        else:
            return sx, sy, sz

    def molecular_field(self):
        return [si[self.bonds].sum(dim=-1) for si in self.spins()]

    def energy(self):
        # sx, sy, sz = self.spins()
        e_list = [s * (s[self.bonds] * self.bond_weights).sum(dim=-1) for s in self.spins()]
        e = e_list[0] + e_list[1] + e_list[2]
        # e = sx * sx[self.bonds].sum(dim=-1) + sy * sy[self.bonds].sum(dim=-1) + sz * sz[self.bonds].sum(dim=-1)
        return e.sum() / 2.


    def plot(self,
             z: Optional[int] = None,
             draw_spins: bool = True,
             draw_lattice: bool = False,
             draw_chirality: bool = False,
             draw_energy: bool = False,
             distortion: bool = False):

        if draw_energy and draw_chirality:
            raise ValueError('You can not draw energy and chirality simultaneously.')

        # fig = plt.figure(figsize=(45, 30), dpi=300)
        plt.figaspect(1 + 1/np.sqrt(2))
        fig = plt.gcf()

        # fig = plt.figure(dpi=100)
        z0 = self.H // 2 if z is None else z

        spin_inds = self.spin_inds_gl.numpy()
        xs, ys, zs = [component[spin_inds] for component in self.xyz_from_inds_gl(np.arange(self.N))]
        z_mask = zs == z0
        spin_inds = np.arange(self.n_spins)[z_mask]
        n_spins = len(spin_inds)

        xs_s = self.xs + (self.ys * 0.5)
        ys_s = self.ys * np.sqrt(2) / 2

        xs, ys = xs_s[z_mask], ys_s[z_mask]

        if distortion:
            xs = xs.astype(float) + np.random.rand(xs.shape[0]) * 0.2
            ys = ys.astype(float) + np.random.rand(xs.shape[0]) * 0.2

        plt.scatter(xs, ys, marker='.')

        if draw_lattice:
            for i1 in spin_inds:
                for i2 in self.bonds_np[i1, [1, 2, 4]]:
                     plt.plot(xs_s[[i1, i2]], ys_s[[i1, i2]], color=(0,0,0,0.2))


        spins = np.array([s.data.cpu().numpy() for s in self.spins()])
        if draw_spins:
            for i in spin_inds:
                plt.arrow(x=xs_s[i] - spins[0, i] / 4,
                          y=ys_s[i] - spins[1, i] / 4,
                          dx=spins[0, i] / 2,
                          dy=spins[1, i] / 2,
                          # width=0.2,
                          length_includes_head=True,
                          # head_starts_at_zero=True,
                          head_width=0.2, head_length=0.5*np.sqrt(spins[0, i]**2 + spins[1, i]**2),
                          color=(max(0, spins[2, i]) ** 0.5, 0, max(0, -spins[2, i]) ** 0.5))


        L_points = fig.get_figwidth() * 72
        L_scale = self.L * (1 + np.sqrt(2) / 2)

        point_size = L_points / L_scale

        if draw_energy:
            for i1 in spin_inds:
                bonds = self.bonds_np[i1, [1, 2, 4]]
                bonds = bonds[bonds != i1]

                en = (spins[:, i1, None] * spins[:, bonds]).sum(axis=0)

                xs_p = (xs_s[i1] + xs_s[bonds]) / 2
                ys_p = (ys_s[i1] + ys_s[bonds]) / 2

                colors = np.array([np.maximum(0, en),
                                   [0]*len(bonds),
                                   np.maximum(0, -en)])
                plt.scatter(xs_p, ys_p, s=(0.29 * point_size)**2, c=colors.T)

        if draw_chirality:
            for i1 in spin_inds:
                bonds = self.bonds_np[i1, [1, 2, 4]]
                bonds = bonds[bonds != i1]
                chir = np.cross(a=spins[:, i1, None],
                                b=spins[:, bonds],
                                axisa=0,
                                axisb=0)
                xs_p = (xs_s[i1] + xs_s[bonds]) / 2
                ys_p = (ys_s[i1] + ys_s[bonds]) / 2
                colors = np.array([np.maximum(0, chir[:, 2]),
                                   [0]*len(bonds),
                                   np.maximum(0, -chir[:, 2])])
                plt.scatter(xs_p, ys_p, s=(0.29 * point_size)**2, c=colors.T)


    def get_angles_from_spins(self, spins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ shape of spins is (3, N)"""
        thetas = torch.acos(spins[2, :])
        phis = torch.acos(spins[0, :] / torch.sin(thetas)) * spins[1, :].sign()
        return thetas, phis

    def optimize_gd(self, n_steps: int, lr: float = 0.001):
        # opt = torch.optim.Adam(params=[self.thetas, self.phis], lr=lr, betas=(0.9, 0.999))
        from utils.optimizer import MyAdam
        opt = MyAdam(params=[self.thetas, self.phis], lr=lr, betas=(0.9, 0.999))
        for _ in trange(n_steps, desc='GD optimization'):
            opt.zero_grad()
            self.energy().backward()
            self.thetas.grad[self.inds_z_bounds] = 0.
            opt.step()

    def optimize_em(self, n_steps: int, lr: float = 0.5, progress: bool = True):
        with torch.no_grad():
            spins = self.spins()
            spins = torch.stack(spins, dim=0)  # shape = (3, N)

            es = []
            es_den = []
            angs = []

            n_bonds = float(self.bond_weights.sum() / 2)
            for i in trange(n_steps, desc='EM optimization', disable=not progress):
                m_field = (spins[:, self.bonds] * self.bond_weights.view(1, -1, 8)).sum(dim=-1)  # shape = (3, N)
                m_field[2, self.inds_z_bounds] = 0.
                m_field_norm = m_field.norm(dim=0)
                m_field_dir = m_field / (m_field_norm + 1e-10)

                coss = -(spins * m_field_dir).sum(dim=0)
                coss[m_field_norm.view(-1) == 0] = 1.
                angles = coss.acos()
                angs.append(float(angles.max()) * 180 / np.pi)

                e = float((spins * m_field_dir).sum())
                es.append(e)
                es_den.append(e / n_bonds)

                spins = spins * (1 - lr) - m_field_dir * lr
                spins = spins / spins.norm(dim=0)

            thetas, phis = self.get_angles_from_spins(spins)
            self.thetas[:] = thetas
            self.phis[:] = phis

        return es, es_den, angs

    def find_twist(self, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            # spins = torch.stack(self.spins(), dim=0).data.cpu().numpy()
            thetas = self.thetas.data.cpu().numpy()
            phis = self.phis.data.cpu().numpy()

        cos_th = np.cos(thetas)
        sin_th = np.sin(thetas)
        cos_phi = np.cos(phis)
        sin_phi = np.sin(phis)

        sx, sy, sz = sin_th * cos_phi, sin_th * sin_phi, cos_th  # shape = (N,)
        spins = np.stack([sx, sy, sz], axis=0)  # shape = (3, N)

        # -------------------------------------
        # first derivative:

        zeros = np.zeros_like(sx)
        sx1 = np.stack([cos_th * cos_phi, -sy], axis=0)  # shape = (2, N)
        sy1 = np.stack([cos_th * sin_phi, sx], axis=0)
        sz1 = np.stack([-sin_th, zeros], axis=0)

        spins_d1 = np.stack([sx1, sy1, sz1], axis=0)  # shape = (3, 2, N)

        # -------------------------------------
        # second derivative:

        sx2 = np.array([[-sx, -sy1[0]],  # shape = (2, 2, N)
                        [-sy1[0], -sx]])

        sy2 = np.array([[-sy, sx1[0]],
                        [sx1[0], -sy]])

        sz2 = np.array([[-sz, zeros],
                        [zeros, zeros]])

        spins_d2 = np.stack([sx2, sy2, sz2], axis=0)  # shape = (3, 2, 2, N)

        # -------------------------------------
        m = (spins[:, self.bonds_np] * self.bond_weights_np.reshape(1, -1, 8)).sum(axis=-1)  # shape = (3, N)

        mus = np.zeros((3, self.n_spins))  # shape = (3, N)
        mus[2, self.inds_z_bounds] = 1.

        eps_x = np.zeros((3, self.n_spins))  # shape = (3, N)
        eps_x[0, self.inds_z0] = 1.

        eps_y = np.zeros((3, self.n_spins))  # shape = (3, N)
        eps_y[1, self.inds_z0] = 1.

        P_spins_d1 = spins_d1.copy()  # shape = (3, 2, N)
        P_spins_d1[2, :, self.inds_z_bounds] = 0.

        P_m = m.copy()  # shape = (3, N)
        P_m[2, self.inds_z_bounds] = 0.

        J = sparse.lil_matrix((self.n_spins, self.n_spins))
        J[np.arange(self.n_spins).reshape(-1, 1), self.bonds_np] = 1.
        J[np.diag_indices_from(J)] = 0

        # -------------------------------------
        # first equation:

        Jzz = NDSparse(2, 2, self.n_spins, block=J)
        A1 = sum([(Jzz * P_spins_d1[i, ..., None, None]) * spins_d1[i, None, None, ...] for i in range(3)])

        I = NDSparse(2, 2, self.n_spins, block=sparse.eye(self.n_spins, format='coo'))

        P = np.eye(3)[..., None] - mus[None, ...] * mus[:, None, :]
        tmp = np.einsum('ijb, ib, jzwb -> zbw', P, m, spins_d2)

        # A2 = sum([I * spins_d2_T[i, ..., None]) * P_m[i, None, :, None, None] for i in range(3)])
        A2 = I * tmp[..., :, None]

        A = A1 + A2  # left side matrix

        Pe_x = (mus[:, None, :] * eps_x[None, ...] + mus[None, ...] * eps_x[:, None, :])  # shape = (3, 3, n)
        Pe_y = (mus[:, None, :] * eps_y[None, ...] + mus[None, ...] * eps_y[:, None, :])

        # right side
        rhs_x = np.einsum('ijn, in, jzn -> zn', Pe_x, m, spins_d1)
        rhs_y = np.einsum('ijn, in, jzn -> zn', Pe_y, m, spins_d1)


        # -------------------------------------

        M = A.get_matrix()
        r_x = SLA.lsqr(M, rhs_x.reshape(-1))[0].reshape(2, -1)
        r_y = SLA.lsqr(M, rhs_y.reshape(-1))[0].reshape(2, -1)

        # -------------------------------------

        H1 = sum([(Jzz * spins_d1[i, ..., None, None]) * spins_d1[i, None, None, ...] for i in range(3)])
        tmp = np.einsum('ib, izwb -> zbw', m, spins_d2)
        H2 = I * tmp[..., :, None]

        d2H = H1 + H2

        d2Hxx = ((d2H @ r_x) * r_x).sum()
        d2Hxy = ((d2H @ r_x) * r_y).sum()
        d2Hyy = ((d2H @ r_y) * r_y).sum()

        hess = np.array([[d2Hxx, d2Hxy],
                         [d2Hxy, d2Hyy]])

        vals, vecs = np.linalg.eigh(hess)

        dH1 = np.einsum('in, izn, ezn -> e', m, spins_d1, [r_x, r_y])

        if verbose:
            print('Jacobian:')
            print(dH1, '\n')

            print('Hessian:')
            print(hess, '\n')

            print('Values:')
            print(vals, '\n')

            print('Vectors:')
            print(vecs, '\n')


        return vals, vecs, dH1

        # ---------------------------------------------------------------------------------------------


    @torch.no_grad()
    def all_spins(self):
        spins = self.spins(single=True)
        all_spins = torch.zeros(3, self.N, dtype=torch.double, device=self._device)
        all_spins[:, self.spin_inds_gl] = spins
        all_spins = all_spins.view(3, self.L, self.L, self.H)
        return all_spins

    def get_fourier(self):
        spins = self.all_spins()
        spins_fft = torch.rfft(spins, signal_ndim=3, onesided=False).pow(2).sum(dim=(0, -1))
        return spins_fft

    @torch.no_grad()
    def get_chirality(self):
        bonds = self.bonds[:, [1, 2, 4]]
        bond_weights = self.bond_weights[:, [1, 2, 4]]

        n_chiral_bonds = int(bond_weights.sum())

        spins = self.spins(single=True)
        spins_nn = spins[:, bonds] * bond_weights[None, ...]  # shape (3_xyz, N, 3_nn)

        spins = spins.cpu().data.numpy()
        spins_nn = spins_nn.cpu().numpy()

        chirality = np.cross(spins[..., None], spins_nn, axis=0).sum(axis=(1, 2)) / n_chiral_bonds  # shape = (3,)

        return np.linalg.norm(chirality)





    def check_minimum(self):
        with torch.no_grad():
            thetas = self.thetas.data.cpu().numpy()
            phis = self.phis.data.cpu().numpy()

        cos_th = np.cos(thetas)
        sin_th = np.sin(thetas)
        cos_phi = np.cos(phis)
        sin_phi = np.sin(phis)

        sx, sy, sz = sin_th * cos_phi, sin_th * sin_phi, cos_th  # shape = (N,)
        spins = np.stack([sx, sy, sz], axis=0)  # shape = (3, N)

        # -------------------------------------
        # first derivative:

        zeros = np.zeros_like(sx)
        sx1 = np.stack([cos_th * cos_phi, -sy], axis=0)  # shape = (2, N)
        sy1 = np.stack([cos_th * sin_phi, sx], axis=0)
        sz1 = np.stack([-sin_th, zeros], axis=0)

        spins_d1 = np.stack([sx1, sy1, sz1], axis=0)  # shape = (3, 2, N)


        mf = (spins[:, self.bonds_np] * self.bond_weights_np[None, ...]).sum(axis=-1)  # shape = (3, N)
        mf[2, self.inds_z_bounds] = 0

        dd = np.einsum('in, izn ->zn', mf, spins_d1)

        print('')



