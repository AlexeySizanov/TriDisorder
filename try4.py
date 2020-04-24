from worker3d import *
import torch

seed = 3
np.random.seed(seed)
torch.manual_seed(seed)

w = Worker3D(L=30, H=30, c=0.6)
w._make_one_realization(lr=0.5, progress=True, device='cpu')

