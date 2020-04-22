from system3d import *
import time

np.random.seed(0)
torch.manual_seed(0)

#%%
s = TDSystem3D(L=10, H=10, c=0.3, device='cpu')
s.make_spins(rand_scale=0.2)

s.optimize(n_steps=3000)

# s.check_minimum()

t0 = time.time()
s.find_twist()
dt = time.time() - t0

print(f'time = {dt:.1f}s')
