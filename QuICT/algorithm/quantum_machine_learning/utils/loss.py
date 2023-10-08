"""Commonly used loss functions."""


from abc import ABC, abstractmethod

import autograd.numpy as np
from autograd import grad
from sympy import *
from typing import Union

from QuICT.core.gate import Variable


class Loss:
    """The Loss class.

    Args:
        item (Union[np.float64, float, int]): The value of loss.
        grads (np.ndarray): The gradients of loss.
    """

    @property
    def item(self) -> Union[np.float64, float, int]:
        """Get loss value.

        Returns:
            Union[np.float64, float, int]: The loss value.
        """
        return self._item

    @property
    def grads(self) -> np.ndarray:
        """Get loss gradients.

        Returns:
            np.ndarray: The loss gradients.
        """
        return self._grads

    def __init__(self, item: Union[np.float64, float, int], grads: np.ndarray):
        """Initialize a Loss instance."""
        self._item = item
        self._grads = grads

    def __str__(self):
        return "Loss(item={}, grads={})".format(self.item, self.grads)


class LossFun(ABC):
    """Base class for loss functions.

    Note:
        User-defined loss functions also need to inherit this class.
    """

    def __init__(self):
        """Initialize a LossFun instance."""
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
    r"""The Hinge Loss.

    $$
    L_{Hinge}(y) = max(0, 1 - \hat{y}y)
    $$
    """

    def __init__(self):
        """Initialize a HingeLoss instance."""
        super().__init__()

    def _get_loss(self, pred: np.ndarray, target: np.ndarray):
        loss = np.clip(1 - pred * target, a_min=0.0, a_max=None)
        return np.mean(loss)

    def __str__(self):
        return "HingeLoss"


class MSELoss(LossFun):
    r"""The Mean Squared Error Loss.

    $$
    L_{MSE}(y) = \frac{1}{N} \sum_{i=0}^{N-1} (\hat{y}_i - y_i)^2
    $$
    """

    def __init__(self):
        """Initialize an MSELoss instance."""
        super().__init__()

    def _get_loss(self, pred: np.ndarray, target: np.ndarray):
        loss = (pred - target) ** 2
        return np.mean(loss)

    def __str__(self):
        return "MSELoss"


class BCELoss(LossFun):
    r"""Compute the Binary Cross Entropy Loss.

    $$
    L_{BCE}(y) = -\frac{1}{N} (y \cdot log(\hat{y}) + (1-y) \cdot log(1 - \hat{y}))
    $$

    Note:
        Target $y$ should be numbers between 0 and 1.

        BCELoss clamps its log function outputs to be greater than
        or equal to -100 to avoid an infinite term in the loss equation.
    """

    def __init__(self):
        """Initialize a BCELoss instance."""
        super().__init__()

    def _get_loss(self, pred: np.ndarray, target: np.ndarray):
        # loss = -target * np.log(pred + 1e-12) - (1 - target) * np.log(1 - pred + 1e-12)
        loss = np.clip(-target * np.log(pred) - (1 - target) * np.log(1 - pred), 0, 100)
        return np.mean(loss)

    def __str__(self):
        return "BCELoss"
