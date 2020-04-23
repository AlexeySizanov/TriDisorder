from system3d import *
import time

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

#%%
s = TDSystem3D(L=20, H=20, c=0.3, device='cpu')
s.make_spins(rand_scale=0.2)

# s.optimize_em(n_steps=1000, lr=0.5)
# print('\n', s.check_minimum())

# s.optimize_em(n_steps=2000, lr=0.25)
# print('\n', s.check_minimum())

s.optimize_em(n_steps=3000, lr=0.4)
print('\n', s.check_minimum())
