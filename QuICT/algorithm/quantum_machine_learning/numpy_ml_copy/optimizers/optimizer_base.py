from copy import deepcopy
from abc import ABC, abstractmethod
from QuICT.core.gate.utils.variable import Variable
import numpy as np
class OptimizerBase(ABC):
    def __init__(self, lr, scheduler=None):
        """
        An abstract base class for all Optimizer objects.

        This should never be used directly.
        """
        from ..initializers import SchedulerInitializer

        self.cache = {}
        self.cur_step = 0
        self.hyperparameters = {}
        self.lr_scheduler = SchedulerInitializer(scheduler, lr=lr)()

    def __call__(self, param, param_grad, param_name, cur_loss=None):
        return self.update(param, param_grad, param_name, cur_loss)

    def step(self):
        """Increment the optimizer step counter by 1"""
        self.cur_step += 1

    def reset_step(self):
        """Reset the step counter to zero"""
        self.cur_step = 0

    def copy(self):
        """Return a copy of the optimizer object"""
        return deepcopy(self)

    def set_params(self, hparam_dict=None, cache_dict=None):
        """Set the parameters of the optimizer object from a dictionary"""
        from ..initializers import SchedulerInitializer

        if hparam_dict is not None:
            for k, v in hparam_dict.items():
                if k in self.hyperparameters:
                    self.hyperparameters[k] = v
                    if k == "lr_scheduler":
                        self.lr_scheduler = SchedulerInitializer(v, lr=None)()

        if cache_dict is not None:
            for k, v in cache_dict.items():
                if k in self.cache:
                    self.cache[k] = v

    @abstractmethod
    def update(self, param, param_grad, param_name, cur_loss=None):
        raise NotImplementedError

