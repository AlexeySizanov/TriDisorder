from worker3d import *

w = Worker3D(L=20, H=20, c=0.6)
w._make_one_realization(progress=True, device='cuda')

