"""Commonly used loss functions."""


from abc import ABC, abstractmethod

import autograd.numpy as np
from autograd import grad
from sympy import *
from typing import Union

from QuICT.core.gate import Variable


class Loss:
    @property
    def item(self):
        return self._item

    @property
    def grads(self):
        return self._grads

    def __init__(self, item: Union[np.float64, float, int], grads: np.ndarray):
        self._item = item
        self._grads = grads

    def __str__(self):
        return "Loss(item={}, grads={})".format(self.item, self.grads)


class LossFun(ABC):
    """The abstract class for loss."""

    def __init__(self):
        self._pred = None
        self._target = None

    def __call__(self, pred: Variable, target: np.array):
        """Call the loss function.

        Args:
            pred (Variable): The predicted values.
            target (np.ndarray): The ground truth.

        Returns:
            Loss: The loss.
        """
        assert pred is not None and target is not None
        assert pred.shape == target.shape
        self._pred = pred
        self._target = target
        loss_item = self._get_loss(pred.pargs, target)
        loss_grads = self._get_grads()
        return Loss(loss_item, loss_grads)

    @abstractmethod
    def _get_loss(self, pred: np.ndarray, target: np.ndarray):
        return NotImplementedError

    def _get_grads(self):
        """Calculate the gradient of the loss function to the predicted values."""
        fun_grad = grad(self._get_loss)
        grads = fun_grad(self._pred.pargs, self._target)
        grads[abs(self._pred.pargs - self._target) < 1e-12] = 0
        idx = abs(self._pred.grads) > 1e-12
        grads[idx] *= self._pred.grads[idx]
        return grads


class HingeLoss(LossFun):
    """Compute the Hinge Loss."""

    def __init__(self):
        super().__init__()

    def _get_loss(self, pred: np.ndarray, target: np.ndarray):
        loss = np.clip(1 - pred * target, a_min=0.0, a_max=None)
        return np.mean(loss)

    def __str__(self):
        return "HingeLoss"


class MSELoss(LossFun):
    """Compute the Mean Squared Error Loss."""

    def __init__(self):
        super().__init__()

    def _get_loss(self, pred: np.ndarray, target: np.ndarray):
        loss = (pred - target) ** 2
        return np.mean(loss)

    def __str__(self):
        return "MSELoss"


class BCELoss(LossFun):
    """Compute the Binary Cross Entropy Loss.

    **Note that the target y should be numbers between 0 and 1.**
    """

    def __init__(self):
        super().__init__()

    def _get_loss(self, pred: np.ndarray, target: np.ndarray):
        loss = -target * np.log(pred + 1e-12) - (1 - target) * np.log(1 - pred + 1e-12)
        return np.mean(loss)

    def __str__(self):
        return "BCELoss"
