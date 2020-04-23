from system3d import *
import time

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

#%%
s = TDSystem3D(L=20, H=20, c=0.5, device='cpu')
s.make_spins(rand_scale=0.2)
print(s.check_minimum())

s.optimize_em(n_steps=3000, lr=0.5)
print('\n', s.check_minimum())

s.optimize_em(n_steps=3000, lr=0.5)
print('\n', s.check_minimum())

s.optimize_em(n_steps=3000, lr=0.5)
print('\n', s.check_minimum())


# es = s.optimize_gd(n_steps=10000, lr=0.001)
# print('\n', s.check_minimum())
# plt.plot(es)
# plt.show()


# s.fine_tune(verbose=True)
# print('\n', s.check_minimum())
