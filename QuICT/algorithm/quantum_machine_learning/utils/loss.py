from sympy import *
import autograd.numpy as np
from autograd import grad


class Loss:
    def __init__(self):
        return

    def __call__(self, pred, true):
        raise NotImplementedError

    def gradient(self, pred, true):
        raise NotImplementedError


class Hinge(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, pred, true):
        loss = np.clip(1 - pred * true, a_min=0.0, a_max=None)
        return np.mean(loss)

    def gradient(self, pred, true):
        fun_grad = grad(self.__call__)
        gradient = fun_grad(pred, true)
        return gradient
