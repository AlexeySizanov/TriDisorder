import numpy as np
import torch
import tqdm
from tqdm import trange
from torch import optim
from system3d import *
from optimizer import DirOpt

np.random.seed(0)
torch.manual_seed(0)

#%%
s = TDSystem3D(L=10, H=10, c=0.3, device='cpu')
s.make_spins(rand_scale=0.2)

#%%
plt.figure(dpi=300)
s.plot(z=0, draw_lattice=True, draw_spins=True, draw_chirality=True)
plt.show()

#%%
opts = [
        optim.Adam(params=[s.thetas, s.phis], lr=0.001, betas=(0.9, 0.99)),
        optim.Adam(params=[s.thetas, s.phis], lr=0.0001, betas=(0.9, 0.99)),
        optim.Adam(params=[s.thetas, s.phis], lr=0.00001, betas=(0.9, 0.99)),
        # optim.SGD(params=[s.thetas, s.phis], lr=10.)
]
# opt = optim.Adam(params=[s.thetas, s.phis], lr=0.001, betas=(0.9, 0.99))
# opt = optim.SGD(params=[s.thetas, s.phis], lr=10.)
# opt = DirOpt(params=[s.thetas, s.phis], lr=10., decay=0.3)

#%%
es = []
es_checkpoints = [float(s.energy())]
for opt in opts:
    for _ in trange(3000):
        opt.zero_grad()
        e = s.energy()
        e.backward()
        s.thetas.grad[s.inds_z_bounds] = 0.

        # s.thetas.grad.data[:] = 0
        # opt.step(loss=float(e))
        opt.step()
        es.append(float(e))
    es_checkpoints.append(es[-1])

# %%
s.plot(z=0, draw_lattice=True, draw_spins=True, draw_chirality=True)
plt.show()

#%%
plt.figure()
plt.plot(es[-4000:])
# plt.plot(es[:])
plt.show()

#%%
for e in es_checkpoints:
    print(e)