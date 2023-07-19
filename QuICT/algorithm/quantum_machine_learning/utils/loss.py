"""Commonly used loss functions."""


from abc import ABC, abstractmethod
from sympy import *
import autograd.numpy as np
from autograd import grad


class Loss(ABC):
    """The abstract class for loss."""

    def __init__(self):
        self._pred = None
        self._target = None

    @abstractmethod
    def __call__(self, pred, target):
        raise NotImplementedError

    def gradient(self):
        """Calculate the gradient of the loss function to the predicted values."""
        assert (
            self._pred is not None and self._target is not None
        ), "Must call loss function first."
        fun_grad = grad(self.__call__)
        gradient = fun_grad(self._pred, self._target)
        gradient[self._pred - self._target == 0] = 0
        return gradient


class HingeLoss(Loss):
    """Compute the Hinge Loss."""

    def __init__(self):
        super().__init__()

    def __call__(self, pred, target):
        """Call the hinge loss.

        Args:
            pred (np.ndarry): The predicted values.
            target (np.ndarry): The ground truth.

        Returns:
            np.float: The hinge loss.
        """
        self._pred = pred
        self._target = target
        loss = np.clip(1 - pred * target, a_min=0.0, a_max=None)
        return np.mean(loss)


class MSELoss(Loss):
    """Compute the Mean Squared Error Loss."""

    def __init__(self):
        super().__init__()

    def __call__(self, pred, target):
        """Call the MSE loss.

        Args:
            pred (np.ndarry): The predicted values.
            target (np.ndarry): The ground truth.

        Returns:
            np.float: The MSE loss.
        """
        self._pred = pred
        self._target = target
        loss = (pred - target) ** 2
        return np.mean(loss)


class BCELoss(Loss):
    """Compute the Binary Cross Entropy Loss.
    
    **Note that the target y should be numbers between 0 and 1.**
    """

    def __init__(self):
        super().__init__()

    def __call__(self, pred, target):
        """Call the Binary Cross Entropy loss.

        Args:
            pred (np.ndarry): The predicted values.
            target (np.ndarry): The ground truth.

        Returns:
            np.float: The BCE loss.
        """
        self._pred = pred
        self._target = target
        loss = -target * np.log(pred + 1e-12) - (1 - target) * np.log(1 - pred + 1e-12)
        return np.mean(loss)
