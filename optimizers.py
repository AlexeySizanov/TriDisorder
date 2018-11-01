from torch.optim import Optimizer, SGD, Adam, Adadelta
import torch

class Medo:
    def __init__(self, angles, thetas, hole_inds):
        self.angles = angles
        self.dir = torch.zeros_like(angles)

    def zero_grad(self):
        if self.var.grad is not None:
            self.var.grad.zero_()
