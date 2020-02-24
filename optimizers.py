from torch.optim import Optimizer, SGD, Adam, Adadelta
from torch.optim.lr_scheduler import LambdaLR
import torch, math

from torch import optim



class Medo:
    def __init__(self, angles, thetas, hole_inds):
        self.angles = angles
        self.dir = torch.zeros_like(angles)

    def zero_grad(self):
        if self.var.grad is not None:
            self.var.grad.zero_()


def LogCosineScheduler(optimizer, lr_min, lr_max, period):
    alpha = math.log(lr_max / lr_min)
    return LambdaLR(optimizer=optimizer,
                    lr_lambda=lambda i: math.exp(alpha * (math.cos(2 * math.pi * i / period) - 1)))