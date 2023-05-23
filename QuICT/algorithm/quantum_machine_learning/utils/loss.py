from abc import ABC, abstractmethod
from sympy import *
import autograd.numpy as np
from autograd import grad


class Loss(ABC):
    def __init__(self):
        self._pred = None
        self._target = None

    @abstractmethod
    def __call__(self, pred, target):
        raise NotImplementedError

    @abstractmethod
    def gradient(self):
        raise NotImplementedError


class HingeLoss(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, pred, target):
        self._pred = pred
        self._target = target
        loss = np.clip(1 - pred * target, a_min=0.0, a_max=None)
        return np.mean(loss)

    def gradient(self):
        assert (
            self._pred is not None and self._target is not None
        ), "Must call loss function first."
        fun_grad = grad(self.__call__)
        gradient = fun_grad(self._pred, self._target)
        return gradient
