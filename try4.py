import torch
import numpy as np

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

import matplotlib.pyplot as plt

from worker3d import Worker3D
from system3d import  TDSystem3D

s = TDSystem3D(L=10, H=10, c=0.5, device='cpu', random_scale=0)

# for i in range(10):
s.optimize_em(n_steps=10000, force_plane=True)
print('')
print(s.check_minimum())
print('')

s.find_twist(verbose=True)

print(float(s.energy()))

# s.plot(z=4, draw_lattice=True)
# plt.show()