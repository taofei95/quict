from copy import deepcopy
from abc import ABC, abstractmethod
from QuICT.core.utils.variable import Variable
import numpy as np
from numpy.linalg import norm
from .optimizer_base import OptimizerBase
class Adam(OptimizerBase):
    def __init__(
        self,
        lr=0.001,
        decay1=0.9,
        decay2=0.999,
        eps=1e-7,
        clip_norm=None,
        lr_scheduler=None,
        **kwargs
    ):
        """
        Adam (adaptive moment estimation) optimization algorithm.

        Notes
        -----
        Designed to combine the advantages of :class:`AdaGrad`, which works
        well with sparse gradients, and :class:`RMSProp`, which works well in
        online and non-stationary settings.

        Parameters
        ----------
        lr : float
            Learning rate for update. This parameter is ignored if using
            :class:`~numpy_ml.neural_nets.schedulers.NoamScheduler`.
            Default is 0.001.
        decay1 : float
            The rate of decay to use for in running estimate of the first
            moment (mean) of the gradient. Default is 0.9.
        decay2 : float
            The rate of decay to use for in running estimate of the second
            moment (variance) of the gradient. Default is 0.999.
        eps : float
            Constant term to avoid divide-by-zero errors during the update
            calc. Default is 1e-7.
        clip_norm : float
            If not None, all param gradients are scaled to have maximum l2 norm of
            `clip_norm` before computing update. Default is None.
        lr_scheduler : str, or :doc:`Scheduler <numpy_ml.neural_nets.schedulers>` object, or None
            The learning rate scheduler. If None, use a constant learning rate
            equal to `lr`. Default is None.
        """
        super().__init__(lr, lr_scheduler)

        self.cache = {}
        self.hyperparameters = {
            "id": "Adam",
            "lr": lr,
            "eps": eps,
            "decay1": decay1,
            "decay2": decay2,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        H = self.hyperparameters
        lr, d1, d2 = H["lr"], H["decay1"], H["decay2"]
        eps, cn, sc = H["eps"], H["clip_norm"], H["lr_scheduler"]
        return "Adam(lr={}, decay1={}, decay2={}, eps={}, clip_norm={}, lr_scheduler={})".format(
            lr, d1, d2, eps, cn, sc
        )

    def update(self, param:Variable, param_name, cur_loss=None):
        """
        Compute the Adam update for a given parameter.

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
            :class:`~numpy_ml.neural_nets.schedulers.KingScheduler`. Default is
            None.

        Returns
        -------
        updated_params : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of `param` after applying the Adam update.
        """
        C = self.cache
        H = self.hyperparameters
        d1, d2 = H["decay1"], H["decay2"]
        eps, clip_norm = H["eps"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        if param_name not in C:
            C[param_name] = {
                "t": 0,
                "mean": np.zeros_like(param.grads),
                "var": np.zeros_like(param.grads),
            }

        # scale gradient to avoid explosion
        t = np.inf if clip_norm is None else clip_norm
        if norm(param.grads) > t:
            param.grads = param.grads * t / norm(param.grads)

        t = C[param_name]["t"] + 1
        var = C[param_name]["var"]
        mean = C[param_name]["mean"]

        # update cache
        C[param_name]["t"] = t
        C[param_name]["var"] = d2 * var + (1 - d2) * param.grads** 2
        C[param_name]["mean"] = d1 * mean + (1 - d1) * param.grads
        self.cache = C

        # calc unbiased moment estimates and Adam update
        v_hat = C[param_name]["var"] / (1 - d2 ** t)
        m_hat = C[param_name]["mean"] / (1 - d1 ** t)
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        #param.pargs = param.pargs - update
        param.pargs -= update
