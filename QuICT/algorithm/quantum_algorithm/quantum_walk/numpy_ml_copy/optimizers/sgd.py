from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
from numpy.linalg import norm
from .optimizer_base import OptimizerBase
class SGD(OptimizerBase):
    def __init__(
        self, lr=0.01, momentum=0.0, clip_norm=None, lr_scheduler=None, **kwargs
    ):
        """
        A stochastic gradient descent optimizer.

        Notes
        -----
        For model parameters :math:`\\theta`, averaged parameter gradients
        :math:`\\nabla_{\\theta} \mathcal{L}`, and learning rate :math:`\eta`,
        the SGD update at timestep `t` is

        .. math::

            \\text{update}^{(t)}
                &=  \\text{momentum} \cdot \\text{update}^{(t-1)} + \eta^{(t)} \\nabla_{\\theta} \mathcal{L}\\\\
            \\theta^{(t+1)}
                &\leftarrow  \\theta^{(t)} - \\text{update}^{(t)}

        Parameters
        ----------
        lr : float
            Learning rate for SGD. If scheduler is not None, this is used as
            the starting learning rate. Default is 0.01.
        momentum : float in range [0, 1]
            The fraction of the previous update to add to the current update.
            If 0, no momentum is applied. Default is 0.
        clip_norm : float
            If not None, all param gradients are scaled to have maximum l2 norm of
            `clip_norm` before computing update. Default is None.
        lr_scheduler : str, :doc:`Scheduler <numpy_ml.neural_nets.schedulers>` object, or None
            The learning rate scheduler. If None, use a constant learning
            rate equal to `lr`. Default is None.
        """
        super().__init__(lr, lr_scheduler)

        self.hyperparameters = {
            "id": "SGD",
            "lr": lr,
            "momentum": momentum,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        H = self.hyperparameters
        lr, mm, cn, sc = H["lr"], H["momentum"], H["clip_norm"], H["lr_scheduler"]
        return "SGD(lr={}, momentum={}, clip_norm={}, lr_scheduler={})".format(
            lr, mm, cn, sc
        )

    def update(self, param, param_grad, param_name, cur_loss=None):
        """
        Compute the SGD update for a given parameter

        Parameters
        ----------
        param : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of the parameter to be updated.
        param_grad : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The gradient of the loss function with respect to `param_name`.
        param_name : str
            The name of the parameter.
        cur_loss : float
            The training or validation loss for the current minibatch. Used for
            learning rate scheduling e.g., by
            :class:`~numpy_ml.neural_nets.schedulers.KingScheduler`.
            Default is None.

        Returns
        -------
        updated_params : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of `param` after applying the momentum update.
        """
        C = self.cache
        H = self.hyperparameters
        momentum, clip_norm = H["momentum"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # scale gradient to avoid explosion
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        update = momentum * C[param_name] + lr * param_grad
        self.cache[param_name] = update
        return param - update
