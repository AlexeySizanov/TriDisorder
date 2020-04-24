import numpy as np
from tqdm import tqdm
from worker3d import Worker3D

L = 20
N = 100
cs = np.linspace(0.4, 0.75, 8)
print(cs)

for c in tqdm(cs, 'Concentrations', ncols=120):
    w = Worker3D(L=L, H=L, c=c)
    w.do(n=N, save_every=10, device='cuda', folder='res', prefix='')