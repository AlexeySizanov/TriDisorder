from system3d import *
import time

np.random.seed(0)
torch.manual_seed(0)

#%%
s = TDSystem3D(L=12, H=12, c=0.3, device='cpu')
s.make_spins(rand_scale=0.2)

s.optimize_em(n_steps=3000)
with torch.no_grad():
    print('E =', float(s.energy()))
s.find_twist(verbose=True)

# es = []
#
# opt = torch.optim.Adam(params=[s.thetas, s.phis], lr=0.001, betas=(0.9, 0.999))
# for _ in trange(10000):
#     opt.zero_grad()
#     e = s.energy()
#     es.append(float(e))
#     e.backward()
#     opt.step()

s.optimize_gd(n_steps=1000)
with torch.no_grad():
    print('E =', float(s.energy()))
s.find_twist(verbose=True)

#%%
