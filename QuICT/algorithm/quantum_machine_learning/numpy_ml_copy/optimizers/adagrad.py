from copy import deepcopy
from abc import ABC, abstractmethod
from QuICT.core.gate.utils.variable import Variable
import numpy as np
from numpy.linalg import norm
from .optimizer_base import OptimizerBase
class AdaGrad(OptimizerBase):
    def __init__(self, lr=0.01, eps=1e-7, clip_norm=None, lr_scheduler=None, **kwargs):
        """
        An AdaGrad optimizer.

        Notes
        -----
        Weights that receive large gradients will have their effective learning
        rate reduced, while weights that receive small or infrequent updates
        will have their effective learning rate increased.

        Equations::

            cache[t] = cache[t-1] + grad[t] ** 2
            update[t] = lr * grad[t] / (np.sqrt(cache[t]) + eps)
            param[t+1] = param[t] - update[t]

        Note that the ``**`` and `/` operations are elementwise

        "A downside of Adagrad ... is that the monotonic learning rate usually
        proves too aggressive and stops learning too early." [1]

        References
        ----------
        .. [1] Karpathy, A. "CS231n: Convolutional neural networks for visual
           recognition" https://cs231n.github.io/neural-networks-3/

        Parameters
        ----------
        lr : float
            Global learning rate
        eps : float
            Smoothing term to avoid divide-by-zero errors in the update calc.
            Default is 1e-7.
        clip_norm : float or None
            If not None, all param gradients are scaled to have maximum `L2` norm of
            `clip_norm` before computing update. Default is None.
        lr_scheduler : str or :doc:`Scheduler <numpy_ml.neural_nets.schedulers>` object or None
            The learning rate scheduler. If None, use a constant learning
            rate equal to `lr`. Default is None.
        """
        super().__init__(lr, lr_scheduler)

        self.cache = {}
        self.hyperparameters = {
            "id": "AdaGrad",
            "lr": lr,
            "eps": eps,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        H = self.hyperparameters
        lr, eps, cn, sc = H["lr"], H["eps"], H["clip_norm"], H["lr_scheduler"]
        return "AdaGrad(lr={}, eps={}, clip_norm={}, lr_scheduler={})".format(
            lr, eps, cn, sc
        )

    def update(self, param:Variable, param_name, cur_loss=None):
        """
        Compute the AdaGrad update for a given parameter.

        Notes
        -----
        Adjusts the learning rate of each weight based on the magnitudes of its
        gradients (big gradient -> small lr, small gradient -> big lr).

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
            The value of `param` after applying the AdaGrad update
        """
        param_grad = param.grads
        C = self.cache
        H = self.hyperparameters
        eps, clip_norm = H["eps"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # scale gradient to avoid explosion
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        C[param_name] += param_grad ** 2
        update = lr * param_grad / (np.sqrt(C[param_name]) + eps)
        self.cache = C
        param.pargs -= update
