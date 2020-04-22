from typing import Optional, List

import numpy as np
from scipy import sparse


class NDSparse:
    """
    Class for emulation of a sparse array of the shape (n1, block_size, n2, block_size)
    i.e. (n1 x n2) matrix of blocks of the size (block_size x block_size).
    """
    def __init__(self, n1, n2, block_size, mats: Optional[List[List[sparse.spmatrix]]] = None, block=None):
        self.n1 = n1
        self.n2  = n2
        self.block_size = block_size

        if mats is None:
            self.ms = [[sparse.lil_matrix((block_size, block_size)) for i2 in range(n2)] for i1 in range(n1)]
        else:
            self.ms = mats

        if block is not None:
            for i1 in range(self.n1):
                for i2 in range(self.n2):
                    self.ms[i1][i2] = block.copy()


    def __add__(self, other: 'NDSparse'):
        mats = [[self.ms[i1][i2] + other.ms[i1][i2] for i2 in range(self.n2)] for i1 in range(self.n1)]
        return NDSparse(self.n1, self.n2, self.block_size, mats=mats)

    def __radd__(self, other):
        if isinstance(other, NDSparse):
            return self + other
        else:
            mats = [[self.ms[i1][i2] + other for i2 in range(self.n2)] for i1 in range(self.n1)]
            return NDSparse(self.n1, self.n2, self.block_size, mats=mats)

    def __mul__(self, a: np.ndarray):
        if a.ndim != 4:
            raise ValueError('array must have 4 dimensions')

        mats = [[None, None], [None, None]]

        n = 0
        for i1 in range(self.n1):
            for i2 in range(self.n2):
                mats[i1][i2] = self.ms[i1][i2].multiply(a[i1 % a.shape[0], :, i2 % a.shape[2], :]).copy()
                n += 1

        return NDSparse(self.n1, self.n2, self.block_size, mats=mats)

    def __matmul__(self, a: np.ndarray) -> np.ndarray:
        if a.shape[:2] != (self.n2, self.block_size):
            raise ValueError('array mast have 2 dimensions.')

        res = np.zeros((self.n1, self.block_size) + a.shape[2:])

        a = a.reshape(self.n2 * self.block_size, *a.shape[2:])
        ms = [sparse.hstack(m) for m in self.ms]

        for i, m in enumerate(ms):
            res[i] = m @ a

        return res





    def get_matrix(self):
        return sparse.vstack([sparse.hstack(m) for m in self.ms])
