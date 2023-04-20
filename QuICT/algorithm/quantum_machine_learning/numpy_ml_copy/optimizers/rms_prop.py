from copy import deepcopy
from abc import ABC, abstractmethod
from QuICT.core.gate.utils.variable import Variable
import numpy as np
from numpy.linalg import norm
from .optimizer_base import OptimizerBase
class RMSProp(OptimizerBase):
    def __init__(
        self, lr=0.001, decay=0.9, eps=1e-7, clip_norm=None, lr_scheduler=None, **kwargs
    ):
        """
        RMSProp optimizer.

        Notes
        -----
        RMSProp was proposed as a refinement of :class:`AdaGrad` to reduce its
        aggressive, monotonically decreasing learning rate.

        RMSProp uses a *decaying average* of the previous squared gradients
        (second moment) rather than just the immediately preceding squared
        gradient for its `previous_update` value.

        Equations::

            cache[t] = decay * cache[t-1] + (1 - decay) * grad[t] ** 2
            update[t] = lr * grad[t] / (np.sqrt(cache[t]) + eps)
            param[t+1] = param[t] - update[t]

        Note that the ``**`` and ``/`` operations are elementwise.

        Parameters
        ----------
        lr : float
            Learning rate for update. Default is 0.001.
        decay : float in [0, 1]
            Rate of decay for the moving average. Typical values are [0.9,
            0.99, 0.999]. Default is 0.9.
        eps : float
            Constant term to avoid divide-by-zero errors during the update calc. Default is 1e-7.
        clip_norm : float or None
            If not None, all param gradients are scaled to have maximum l2 norm of
            `clip_norm` before computing update. Default is None.
        lr_scheduler : str or :doc:`Scheduler <numpy_ml.neural_nets.schedulers>` object or None
            The learning rate scheduler. If None, use a constant learning
            rate equal to `lr`. Default is None.
        """
        super().__init__(lr, lr_scheduler)

        self.cache = {}
        self.hyperparameters = {
            "id": "RMSProp",
            "lr": lr,
            "eps": eps,
            "decay": decay,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        H = self.hyperparameters
        sc = H["lr_scheduler"]
        lr, eps, dc, cn = H["lr"], H["eps"], H["decay"], H["clip_norm"]
        return "RMSProp(lr={}, eps={}, decay={}, clip_norm={}, lr_scheduler={})".format(
            lr, eps, dc, cn, sc
        )

    def update(self, param:Variable ,param_name, cur_loss=None):
        """
        Compute the RMSProp update for a given parameter.

        Parameters
        ----------
        param : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of the parameter to be updated
        param_grad : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The gradient of the loss function with respect to `param_name`
        param_name : str
            The name of the parameter
        cur_loss : float or None
            The training or validation loss for the current minibatch. Used for
            learning rate scheduling e.g., by
            :class:`~numpy_ml.neural_nets.schedulers.KingScheduler`.
            Default is None.

        Returns
        -------
        updated_params : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of `param` after applying the RMSProp update.
        """
        param_grad = param.grads
        C = self.cache
        H = self.hyperparameters
        eps, decay, clip_norm = H["eps"], H["decay"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # scale gradient to avoid explosion
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        C[param_name] = decay * C[param_name] + (1 - decay) * param_grad ** 2
        update = lr * param_grad / (np.sqrt(C[param_name]) + eps)
        self.cache = C
        param.pargs -= update

