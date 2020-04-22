from worker3d import *

w = Worker3D(L=10, H=10, c=0.3)

w.do(n=3)

print(w.n_realizations)

w = w + w

w.save('aaa')
q = Worker3D.load('aaa')

print('')


