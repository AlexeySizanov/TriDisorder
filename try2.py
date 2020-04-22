from system3d import *

np.random.seed(0)
torch.manual_seed(0)

#%%
s = TDSystem3D(L=10, H=10, c=0.3, device='cpu')
s.make_spins(rand_scale=0.9)

#%%
# plt.figure(dpi=300)
# s.plot(z=0, draw_lattice=True, draw_spins=True, draw_chirality=True)
# plt.show()

es, es_den, angs = s.optimize_em(n_steps=3000, lr=0.5)
# es, es_den, angs = s.optimize(n_steps=1000, lr=0.1)
# es, es_den, angs = s.optimize(n_steps=1000, lr=0.03)


# %%
# s.plot(z=0, draw_lattice=True, draw_spins=True, draw_chirality=True)
# plt.show()

#%%
# plt.figure()
# plt.plot(es[-400:])
# plt.plot(es[:])
# plt.show()


print(es[-10:])